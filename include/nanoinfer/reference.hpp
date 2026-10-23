#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "nanoinfer/model.hpp"

namespace nanoinfer {

struct ForwardResult {
  std::size_t positions;
  std::vector<float> logits;
  std::vector<std::vector<float>> hidden_states;
  std::vector<float> final_ln;
};

// Single-thread reference execution. Every generation step recomputes the
// complete prefix, using absolute position IDs starting at zero; no KV cache.
class ReferenceTransformer {
 public:
  explicit ReferenceTransformer(const std::filesystem::path& path) : model_(path) {
    for (const auto& view : model_.tensors()) {
      if (view.dtype != DType::f32) {
        throw std::invalid_argument("the full-prefix reference needs a float32 model");
      }
    }
  }
  [[nodiscard]] const ModelConfig& config() const noexcept { return model_.config(); }
  [[nodiscard]] ForwardResult forward(std::span<const std::uint32_t> tokens,
                                      bool capture_hidden = true,
                                      bool last_logits_only = false) const;
  [[nodiscard]] std::vector<std::uint32_t> greedy(std::span<const std::uint32_t> prompt,
                                                  std::size_t new_tokens) const;

 private:
  void validate_tokens(std::span<const std::uint32_t> tokens) const;
  Model model_;
};

}  // namespace nanoinfer
