#include "nanoinfer/sampling.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace nanoinfer {
Sampler::Sampler(SamplingOptions options, std::uint64_t seed)
    : options_(options), generator_(seed) {
  if (!std::isfinite(options.temperature) || options.temperature < 0 ||
      !std::isfinite(options.top_p) || options.top_p <= 0 || options.top_p > 1) {
    throw std::invalid_argument(
        "temperature must be finite and nonnegative; top-p must be in (0,1]");
  }
}

std::uint32_t Sampler::sample(std::span<const float> logits) {
  if (logits.empty() || logits.size() > std::numeric_limits<std::uint32_t>::max() ||
      !std::all_of(logits.begin(), logits.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::invalid_argument("sampling requires a nonempty finite logit vector");
  }
  if (options_.temperature == 0) {
    return static_cast<std::uint32_t>(std::max_element(logits.begin(), logits.end()) -
                                      logits.begin());
  }
  std::vector<std::uint32_t> indices(logits.size());
  std::iota(indices.begin(), indices.end(), std::uint32_t{0});
  std::sort(indices.begin(), indices.end(), [logits](auto left, auto right) {
    return logits[left] == logits[right] ? left < right : logits[left] > logits[right];
  });
  if (options_.top_k != 0 && options_.top_k < indices.size()) indices.resize(options_.top_k);
  std::vector<double> weights(indices.size());
  const double maximum = logits[indices.front()];
  double total = 0;
  for (std::size_t i = 0; i < indices.size(); ++i) {
    weights[i] =
        std::exp((static_cast<double>(logits[indices[i]]) - maximum) / options_.temperature);
    total += weights[i];
  }
  // Top-p is applied after top-k and keeps the token that crosses the cutoff.
  if (options_.top_p < 1) {
    const auto threshold = options_.top_p * total;
    double cumulative = 0;
    std::size_t count = 0;
    do {
      cumulative += weights[count++];
    } while (count < weights.size() && cumulative < threshold);
    weights.resize(count);
    indices.resize(count);
    total = cumulative;
  }
  // Define the integer-to-uniform mapping instead of relying on the
  // implementation-specific mapping of std::discrete_distribution.
  const double uniform = static_cast<double>(generator_() >> 11) * 0x1.0p-53;
  const double target = uniform * total;
  double cumulative = 0;
  for (std::size_t i = 0; i < weights.size(); ++i) {
    cumulative += weights[i];
    if (target < cumulative) return indices[i];
  }
  return indices.back();
}
}  // namespace nanoinfer
