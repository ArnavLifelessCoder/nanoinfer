#pragma once

#include <optional>

#include "nanoinfer/gemm.hpp"
#include "nanoinfer/kv_cache.hpp"
#include "nanoinfer/reference.hpp"

namespace nanoinfer {
// One sequence per instance. Calls mutate its cache and are not concurrent.
// Matrix products use the scalar reference unless an optimized GEMM step is
// selected; the reference choice is bit-identical to ReferenceTransformer.
class CachedTransformer {
 public:
  explicit CachedTransformer(const std::filesystem::path& path,
                             std::optional<kernels::GemmStep> gemm = std::nullopt);
  [[nodiscard]] const ModelConfig& config() const noexcept { return model_.config(); }
  [[nodiscard]] std::optional<kernels::GemmStep> gemm_step() const noexcept { return gemm_; }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void reset() noexcept { position_ = 0; }
  [[nodiscard]] ForwardResult prefill(std::span<const std::uint32_t> tokens,
                                      bool capture_hidden = false);
  [[nodiscard]] ForwardResult decode(std::uint32_t token, bool capture_hidden = false);

 private:
  void validate_tokens(std::span<const std::uint32_t> tokens) const;
  void matmul(std::span<const float> a, std::span<const float> b, std::size_t rows,
              std::size_t columns, std::size_t inner, std::span<float> output,
              bool transpose_b = false) const;
  ForwardResult append(std::span<const std::uint32_t> tokens, bool capture_hidden);
  Model model_;
  KVCache cache_;
  std::optional<kernels::GemmStep> gemm_;
  std::size_t position_ = 0;
};
}  // namespace nanoinfer
