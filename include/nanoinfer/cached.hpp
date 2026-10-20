#pragma once

#include "nanoinfer/kv_cache.hpp"
#include "nanoinfer/reference.hpp"

namespace nanoinfer {
// One sequence per instance. Calls mutate its cache and are not concurrent.
class CachedTransformer {
 public:
  explicit CachedTransformer(const std::filesystem::path& path);
  [[nodiscard]] const ModelConfig& config() const noexcept { return model_.config(); }
  [[nodiscard]] std::size_t position() const noexcept { return position_; }
  void reset() noexcept { position_ = 0; }
  [[nodiscard]] ForwardResult prefill(std::span<const std::uint32_t> tokens,
                                      bool capture_hidden = false);
  [[nodiscard]] ForwardResult decode(std::uint32_t token, bool capture_hidden = false);

 private:
  void validate_tokens(std::span<const std::uint32_t> tokens) const;
  ForwardResult append(std::span<const std::uint32_t> tokens, bool capture_hidden);
  Model model_;
  KVCache cache_;
  std::size_t position_ = 0;
};
}  // namespace nanoinfer
