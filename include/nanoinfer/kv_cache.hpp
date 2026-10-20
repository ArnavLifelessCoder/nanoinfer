#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace nanoinfer {

// Each layer has separate contiguous [context, heads, head_width] K/V buffers.
// Storage is allocated once. The caller controls the valid prefix length.
class KVCache {
 public:
  KVCache(std::size_t layers, std::size_t context, std::size_t heads, std::size_t head_width);
  void store(std::size_t layer, std::size_t first_position, std::span<const float> fused_qkv);
  [[nodiscard]] std::span<const float> keys(std::size_t layer) const;
  [[nodiscard]] std::span<const float> values(std::size_t layer) const;

 private:
  struct Layer {
    std::vector<float> keys, values;
  };
  std::size_t context_, width_;
  std::vector<Layer> layers_;
};

namespace ops::ref {
// Queries occupy positions [first_position, first_position + rows).
// K/V spans contain exactly the complete valid prefix, including those rows.
void cached_attention(std::span<const float> fused_qkv, std::span<const float> keys,
                      std::span<const float> values, std::size_t first_position, std::size_t width,
                      std::size_t heads, std::span<float> output);
}  // namespace ops::ref
}  // namespace nanoinfer
