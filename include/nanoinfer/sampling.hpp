#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>

namespace nanoinfer {
struct SamplingOptions {
  double temperature = 0.0;  // Zero selects greedy.
  std::size_t top_k = 0;     // Zero disables top-k.
  double top_p = 1.0;
};

class Sampler {
 public:
  explicit Sampler(SamplingOptions options = {}, std::uint64_t seed = 42);
  [[nodiscard]] std::uint32_t sample(std::span<const float> logits);

 private:
  SamplingOptions options_;
  std::mt19937_64 generator_;
};
}  // namespace nanoinfer
