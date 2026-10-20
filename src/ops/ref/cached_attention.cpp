#include <cmath>
#include <stdexcept>
#include <vector>

#include "nanoinfer/kv_cache.hpp"
#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer::ops::ref {
void cached_attention(std::span<const float> fused_qkv, std::span<const float> keys,
                      std::span<const float> values, std::size_t first_position, std::size_t width,
                      std::size_t heads, std::span<float> output) {
  const auto stride = elements(3, width);
  if (width == 0 || heads == 0 || width % heads != 0 || fused_qkv.empty() ||
      fused_qkv.size() % stride != 0 || keys.size() != values.size() || keys.size() % width != 0) {
    throw std::invalid_argument("cached attention shape mismatch");
  }
  const auto rows = fused_qkv.size() / stride;
  const auto positions = keys.size() / width;
  if (first_position > positions || rows != positions - first_position ||
      output.size() != elements(rows, width)) {
    throw std::invalid_argument("cached attention prefix mismatch");
  }
  const auto head_width = width / heads;
  const auto scale = 1.0F / std::sqrt(static_cast<float>(head_width));
  std::vector<float> scores(positions);
  const auto* qkv = fused_qkv.data();
  const auto* key_data = keys.data();
  const auto* value_data = values.data();
  for (std::size_t head = 0; head < heads; ++head) {
    for (std::size_t row = 0; row < rows; ++row) {
      const auto last_key = first_position + row;
      for (std::size_t key = 0; key <= last_key; ++key) {
        float dot = 0.0F;
        for (std::size_t k = 0; k < head_width; ++k) {
          dot += qkv[row * stride + head * head_width + k] *
                 key_data[key * width + head * head_width + k];
        }
        scores[key] = dot * scale;
      }
      softmax(std::span(scores).first(last_key + 1));
      for (std::size_t k = 0; k < head_width; ++k) {
        float sum = 0.0F;
        for (std::size_t key = 0; key <= last_key; ++key) {
          sum += scores[key] * value_data[key * width + head * head_width + k];
        }
        output[row * width + head * head_width + k] = sum;
      }
    }
  }
}
}  // namespace nanoinfer::ops::ref
