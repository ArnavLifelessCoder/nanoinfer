#include "nanoinfer/cached.hpp"

#include <array>
#include <numeric>
#include <stdexcept>
#include <string>

#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer {
namespace ref = ops::ref;

CachedTransformer::CachedTransformer(const std::filesystem::path& path)
    : model_(path),
      cache_(config().layers, config().ctx, config().heads, config().d_model / config().heads) {}

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
      ref::layer_norm(hidden, weight(prefix + "ln_1.weight"), weight(prefix + "ln_1.bias"),
                      normalized);
      ref::matmul(normalized, weight(prefix + "attn.c_attn.weight"), n, three_d, d, qkv);
      ref::bias(qkv, weight(prefix + "attn.c_attn.bias"));
      cache_.store(layer, position_, qkv);
      ref::cached_attention(qkv, cache_.keys(layer).first(valid_elements),
                            cache_.values(layer).first(valid_elements), position_, d,
                            config().heads, attended);
      ref::matmul(attended, weight(prefix + "attn.c_proj.weight"), n, d, d, projected);
      ref::bias(projected, weight(prefix + "attn.c_proj.bias"));
      ref::residual(hidden, projected);

      ref::layer_norm(hidden, weight(prefix + "ln_2.weight"), weight(prefix + "ln_2.bias"),
                      normalized);
      ref::matmul(normalized, weight(prefix + "mlp.c_fc.weight"), n, four_d, d, expanded);
      ref::bias(expanded, weight(prefix + "mlp.c_fc.bias"));
      ref::gelu(expanded);
      ref::matmul(expanded, weight(prefix + "mlp.c_proj.weight"), n, d, four_d, projected);
      ref::bias(projected, weight(prefix + "mlp.c_proj.bias"));
      ref::residual(hidden, projected);
      if (capture_hidden) result.hidden_states.push_back(hidden);
    }
    ref::layer_norm(hidden, weight("ln_f.weight"), weight("ln_f.bias"), normalized);
    result.positions = n;
    result.logits.resize(ref::elements(n, config().vocab));
    ref::matmul(normalized, weight("wte.weight"), n, config().vocab, d, result.logits, true);
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
