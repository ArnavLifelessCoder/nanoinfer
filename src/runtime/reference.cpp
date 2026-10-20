#include "nanoinfer/reference.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer {
namespace ref = ops::ref;

void ReferenceTransformer::validate_tokens(std::span<const std::uint32_t> tokens) const {
  if (tokens.empty() || tokens.size() > config().ctx) {
    throw std::invalid_argument("prompt must contain between one and ctx tokens");
  }
  for (const auto token : tokens) {
    if (token >= config().vocab) {
      throw std::out_of_range("token ID outside vocabulary");
    }
  }
}

ForwardResult ReferenceTransformer::forward(std::span<const std::uint32_t> tokens,
                                            bool capture_hidden, bool last_logits_only) const {
  validate_tokens(tokens);
  const auto n = tokens.size();
  const std::size_t d = config().d_model;
  const auto nd = ref::elements(n, d);
  const auto three_d = ref::elements(3, d);
  const auto four_d = ref::elements(4, d);
  auto weight = [this](const std::string& name) { return model_.tensor(name).values; };
  std::vector<float> hidden(nd), normalized(nd), projected(nd), attended(nd);
  std::vector<float> qkv(ref::elements(n, three_d)), expanded(ref::elements(n, four_d));
  std::vector<std::uint32_t> positions(n);
  std::iota(positions.begin(), positions.end(), std::uint32_t{0});
  ref::embedding(tokens, weight("wte.weight"), config().vocab, d, hidden);
  ref::embedding(positions, weight("wpe.weight"), config().ctx, d, projected);
  ref::residual(hidden, projected);

  ForwardResult result{};
  for (std::uint32_t layer = 0; layer < config().layers; ++layer) {
    const auto prefix = "h." + std::to_string(layer) + ".";
    ref::layer_norm(hidden, weight(prefix + "ln_1.weight"), weight(prefix + "ln_1.bias"),
                    normalized);
    ref::matmul(normalized, weight(prefix + "attn.c_attn.weight"), n, three_d, d, qkv);
    ref::bias(qkv, weight(prefix + "attn.c_attn.bias"));
    ref::causal_attention(qkv, n, d, config().heads, attended);
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
    if (capture_hidden) {
      result.hidden_states.push_back(hidden);
    }
  }
  ref::layer_norm(hidden, weight("ln_f.weight"), weight("ln_f.bias"), normalized);
  result.positions = last_logits_only ? 1 : n;
  result.logits.resize(ref::elements(result.positions, config().vocab));
  const auto final_input = last_logits_only ? std::span(normalized).last(d) : std::span(normalized);
  // Tied output embedding: wte is [vocab, width], so this product uses its transpose.
  ref::matmul(final_input, weight("wte.weight"), result.positions, config().vocab, d, result.logits,
              true);
  if (capture_hidden) {
    result.final_ln = std::move(normalized);
  }
  return result;
}

std::vector<std::uint32_t> ReferenceTransformer::greedy(std::span<const std::uint32_t> prompt,
                                                        std::size_t new_tokens) const {
  validate_tokens(prompt);
  if (new_tokens > config().ctx - prompt.size()) {
    throw std::invalid_argument("prompt and continuation exceed model context");
  }
  std::vector<std::uint32_t> sequence(prompt.begin(), prompt.end());
  std::vector<std::uint32_t> generated;
  generated.reserve(new_tokens);
  for (std::size_t step = 0; step < new_tokens; ++step) {
    const auto result = forward(sequence, false, true);
    const auto best = std::max_element(result.logits.begin(), result.logits.end());
    const auto token = static_cast<std::uint32_t>(best - result.logits.begin());
    generated.push_back(token);
    sequence.push_back(token);
  }
  return generated;
}
}  // namespace nanoinfer
