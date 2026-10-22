#include "nanoinfer/cached.hpp"

#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>
#include <string>

#include "nanoinfer/attention.hpp"
#include "nanoinfer/gemv.hpp"
#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer {
namespace ref = ops::ref;

CachedTransformer::CachedTransformer(const std::filesystem::path& path,
                                     std::optional<kernels::GemmStep> gemm, std::size_t threads)
    : model_(path),
      cache_(config().layers, config().ctx, config().heads, config().d_model / config().heads),
      gemm_(gemm) {
  if (gemm_ && !kernels::available(*gemm_)) {
    throw std::invalid_argument("requested GEMM kernel is not supported on this CPU");
  }
  if (threads == 0) throw std::invalid_argument("thread count must be positive");
  if (threads > 1) {
    if (!gemm_) throw std::invalid_argument("the scalar reference path is single-threaded");
    pool_ = std::make_unique<ThreadPool>(threads);
  }
}

CachedTransformer::~CachedTransformer() = default;

void CachedTransformer::matmul(std::span<const float> a, std::span<const float> b, std::size_t rows,
                               std::size_t columns, std::size_t inner, std::span<float> output,
                               bool transpose_b) const {
  if (!gemm_) {
    ref::matmul(a, b, rows, columns, inner, output, transpose_b);
  } else if (rows == 1 &&
             (*gemm_ == kernels::GemmStep::packed || *gemm_ == kernels::GemmStep::avx2)) {
    const auto kernel =
        *gemm_ == kernels::GemmStep::avx2 ? kernels::GemvKernel::avx2 : kernels::GemvKernel::scalar;
    kernels::gemv(kernel, a, b, columns, inner, output, transpose_b, pool_.get());
  } else {
    kernels::gemm(*gemm_, a, b, rows, columns, inner, output, transpose_b, pool_.get());
  }
}

void CachedTransformer::layer_norm(std::span<const float> input, std::span<const float> scale,
                                   std::span<const float> shift, std::span<float> output) const {
  const auto width = scale.size();
  const auto rows = width == 0 ? 0 : input.size() / width;
  if (!pool_ || rows < 2) return ref::layer_norm(input, scale, shift, output);
  // Rows are independent; each is normalized exactly as in the reference.
  pool_->parallel_for(rows, [&](std::size_t begin, std::size_t end) {
    ref::layer_norm(input.subspan(begin * width, (end - begin) * width), scale, shift,
                    output.subspan(begin * width, (end - begin) * width));
  });
}

void CachedTransformer::gelu(std::span<float> values) const {
  constexpr std::size_t chunk = 4096;
  if (!pool_ || values.size() <= chunk) return ref::gelu(values);
  pool_->parallel_for((values.size() + chunk - 1) / chunk, [&](std::size_t begin, std::size_t end) {
    const auto first = begin * chunk;
    ref::gelu(values.subspan(first, std::min(values.size(), end * chunk) - first));
  });
}

void CachedTransformer::validate_tokens(std::span<const std::uint32_t> tokens) const {
  if (tokens.empty() || tokens.size() > config().ctx) {
    throw std::invalid_argument("prompt must contain between one and ctx tokens");
  }
  for (const auto token : tokens) {
    if (token >= config().vocab) throw std::out_of_range("token ID outside vocabulary");
  }
}

ForwardResult CachedTransformer::prefill(std::span<const std::uint32_t> tokens,
                                         bool capture_hidden) {
  validate_tokens(tokens);
  reset();
  return append(tokens, capture_hidden);
}

ForwardResult CachedTransformer::decode(std::uint32_t token, bool capture_hidden) {
  if (position_ == 0 || position_ == config().ctx) {
    throw std::invalid_argument("decode requires a nonempty cache with remaining context");
  }
  const std::array<std::uint32_t, 1> tokens{token};
  validate_tokens(tokens);
  return append(tokens, capture_hidden);
}

ForwardResult CachedTransformer::append(std::span<const std::uint32_t> tokens,
                                        bool capture_hidden) {
  try {
    const auto n = tokens.size();
    const std::size_t d = config().d_model;
    const auto nd = ref::elements(n, d);
    const auto three_d = ref::elements(3, d);
    const auto four_d = ref::elements(4, d);
    auto weight = [this](const std::string& name) { return model_.tensor(name).values; };
    std::vector<float> hidden(nd), normalized(nd), projected(nd), attended(nd);
    std::vector<float> qkv(ref::elements(n, three_d)), expanded(ref::elements(n, four_d));
    std::vector<std::uint32_t> positions(n);
    std::iota(positions.begin(), positions.end(), static_cast<std::uint32_t>(position_));
    ref::embedding(tokens, weight("wte.weight"), config().vocab, d, hidden);
    ref::embedding(positions, weight("wpe.weight"), config().ctx, d, projected);
    ref::residual(hidden, projected);

    ForwardResult result{};
    const auto valid_elements = ref::elements(position_ + n, d);
    for (std::uint32_t layer = 0; layer < config().layers; ++layer) {
      const auto prefix = "h." + std::to_string(layer) + ".";
      layer_norm(hidden, weight(prefix + "ln_1.weight"), weight(prefix + "ln_1.bias"), normalized);
      matmul(normalized, weight(prefix + "attn.c_attn.weight"), n, three_d, d, qkv);
      ref::bias(qkv, weight(prefix + "attn.c_attn.bias"));
      cache_.store(layer, position_, qkv);
      const auto keys = cache_.keys(layer).first(valid_elements);
      const auto values = cache_.values(layer).first(valid_elements);
      if (gemm_ == kernels::GemmStep::packed || gemm_ == kernels::GemmStep::avx2) {
        const auto kernel = gemm_ == kernels::GemmStep::avx2 ? kernels::GemvKernel::avx2
                                                             : kernels::GemvKernel::scalar;
        const auto units = kernels::attention_units(n, config().heads);
        const auto attend = [&](std::size_t begin, std::size_t end) {
          kernels::cached_attention(kernel, qkv, keys, values, position_, d, config().heads,
                                    attended, begin, end);
        };
        if (pool_) {
          pool_->parallel_for(units, attend);
        } else {
          attend(0, units);
        }
      } else if (pool_) {
        pool_->parallel_for(config().heads, [&](std::size_t begin, std::size_t end) {
          ref::cached_attention(qkv, keys, values, position_, d, config().heads, attended, begin,
                                end);
        });
      } else {
        ref::cached_attention(qkv, keys, values, position_, d, config().heads, attended);
      }
      matmul(attended, weight(prefix + "attn.c_proj.weight"), n, d, d, projected);
      ref::bias(projected, weight(prefix + "attn.c_proj.bias"));
      ref::residual(hidden, projected);

      layer_norm(hidden, weight(prefix + "ln_2.weight"), weight(prefix + "ln_2.bias"), normalized);
      matmul(normalized, weight(prefix + "mlp.c_fc.weight"), n, four_d, d, expanded);
      ref::bias(expanded, weight(prefix + "mlp.c_fc.bias"));
      gelu(expanded);
      matmul(expanded, weight(prefix + "mlp.c_proj.weight"), n, d, four_d, projected);
      ref::bias(projected, weight(prefix + "mlp.c_proj.bias"));
      ref::residual(hidden, projected);
      if (capture_hidden) result.hidden_states.push_back(hidden);
    }
    layer_norm(hidden, weight("ln_f.weight"), weight("ln_f.bias"), normalized);
    result.positions = n;
    result.logits.resize(ref::elements(n, config().vocab));
    matmul(normalized, weight("wte.weight"), n, config().vocab, d, result.logits, true);
    if (capture_hidden) result.final_ln = std::move(normalized);
    position_ += n;
    return result;
  } catch (...) {
    // A partially completed forward must never leave a cache usable for decode.
    reset();
    throw;
  }
}
}  // namespace nanoinfer
