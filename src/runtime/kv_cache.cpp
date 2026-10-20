#include "nanoinfer/kv_cache.hpp"

#include <algorithm>
#include <stdexcept>

#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer {
KVCache::KVCache(std::size_t layers, std::size_t context, std::size_t heads, std::size_t head_width)
    : context_(context), width_(ops::ref::elements(heads, head_width)) {
  if (layers == 0 || context == 0 || heads == 0 || head_width == 0) {
    throw std::invalid_argument("zero cache dimension");
  }
  const auto size = ops::ref::elements(context_, width_);
  static_cast<void>(ops::ref::elements(ops::ref::elements(layers, size), 2 * sizeof(float)));
  layers_.resize(layers);
  for (auto& layer : layers_) {
    layer.keys.resize(size);
    layer.values.resize(size);
  }
}

void KVCache::store(std::size_t layer, std::size_t first_position,
                    std::span<const float> fused_qkv) {
  auto& destination = layers_.at(layer);
  const auto stride = ops::ref::elements(3, width_);
  if (fused_qkv.empty() || fused_qkv.size() % stride != 0 || first_position > context_ ||
      fused_qkv.size() / stride > context_ - first_position) {
    throw std::invalid_argument("cache write exceeds shape or context");
  }
  for (std::size_t row = 0; row < fused_qkv.size() / stride; ++row) {
    const auto offset = (first_position + row) * width_;
    std::copy_n(fused_qkv.data() + row * stride + width_, width_, destination.keys.data() + offset);
    std::copy_n(fused_qkv.data() + row * stride + 2 * width_, width_,
                destination.values.data() + offset);
  }
}

std::span<const float> KVCache::keys(std::size_t layer) const { return layers_.at(layer).keys; }
std::span<const float> KVCache::values(std::size_t layer) const { return layers_.at(layer).values; }
}  // namespace nanoinfer
