#pragma once

#include <memory>
#include <optional>

#include "nanoinfer/gemm.hpp"
#include "nanoinfer/kv_cache.hpp"
#include "nanoinfer/reference.hpp"
#include "nanoinfer/thread_pool.hpp"

namespace nanoinfer {
// One sequence per instance. Calls mutate its cache and are not concurrent.
// Matrix products use the scalar reference unless an optimized GEMM step is
// selected; the reference choice is bit-identical to ReferenceTransformer.
// With an optimized step, one-row products (decode) use the GEMV, and threads
// > 1 runs the GEMM, GEMV, attention heads, LayerNorm rows and GELU on a
// pinned thread pool. Every parallel split has a fixed order, so the output is
// bit-identical for any thread count.
class CachedTransformer {
 public:
  explicit CachedTransformer(const std::filesystem::path& path,
                             std::optional<kernels::GemmStep> gemm = std::nullopt,
                             std::size_t threads = 1);
  ~CachedTransformer();
  CachedTransformer(const CachedTransformer&) = delete;
  CachedTransformer& operator=(const CachedTransformer&) = delete;
  [[nodiscard]] std::size_t threads() const noexcept { return pool_ ? pool_->size() : 1; }
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
  void layer_norm(std::span<const float> input, std::span<const float> scale,
                  std::span<const float> shift, std::span<float> output) const;
  void gelu(std::span<float> values) const;
  Model model_;
  KVCache cache_;
  std::optional<kernels::GemmStep> gemm_;
  std::unique_ptr<ThreadPool> pool_;
  std::size_t position_ = 0;
};
}  // namespace nanoinfer
