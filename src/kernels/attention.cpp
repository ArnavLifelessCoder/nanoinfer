#include "nanoinfer/attention.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "kernels/vector_kernels.hpp"
#include "nanoinfer/ref_ops.hpp"

namespace nanoinfer::kernels {
namespace {
void axpy_scalar(float* y, const float* source, float x, std::size_t count) {
  for (std::size_t j = 0; j < count; ++j) y[j] += x * source[j];
}

float dot_scalar(const float* a, const float* b, std::size_t count) {
  float total = 0.0F;
  for (std::size_t i = 0; i < count; ++i) total += a[i] * b[i];
  return total;
}
}  // namespace

std::size_t attention_units(std::size_t rows, std::size_t heads) noexcept {
  return heads * ((rows + kAttentionRows - 1) / kAttentionRows);
}

void cached_attention(GemvKernel kernel, std::span<const float> fused_qkv,
                      std::span<const float> keys, std::span<const float> values,
                      std::size_t first_position, std::size_t width, std::size_t heads,
                      std::span<float> output, std::size_t unit_begin, std::size_t unit_end) {
  const auto stride = ops::ref::elements(3, width);
  if (width == 0 || heads == 0 || width % heads != 0 || fused_qkv.empty() ||
      fused_qkv.size() % stride != 0 || keys.size() != values.size() || keys.size() % width != 0) {
    throw std::invalid_argument("cached attention shape mismatch");
  }
  const auto rows = fused_qkv.size() / stride;
  const auto positions = keys.size() / width;
  if (first_position > positions || rows != positions - first_position ||
      output.size() != ops::ref::elements(rows, width) || unit_begin > unit_end ||
      unit_end > attention_units(rows, heads)) {
    throw std::invalid_argument("cached attention prefix or unit mismatch");
  }
  if (!available(kernel)) throw std::runtime_error("the AVX2 attention needs AVX2 and FMA");
#if defined(NANOINFER_HAVE_AVX2_KERNEL)
  const bool vector = kernel == GemvKernel::avx2;
  const auto axpy = vector ? detail::axpy_avx2 : axpy_scalar;
  const auto dot = vector ? detail::dot_avx2 : dot_scalar;
#else
  const auto axpy = axpy_scalar;
  const auto dot = dot_scalar;
#endif
  const auto head_width = width / heads;
  const auto chunks = (rows + kAttentionRows - 1) / kAttentionRows;
  const auto scale = 1.0F / std::sqrt(static_cast<float>(head_width));
  thread_local std::vector<float> scores;
  scores.resize(positions);
  for (auto unit = unit_begin; unit < unit_end; ++unit) {
    const auto head = unit / chunks;
    const auto row_end = std::min(rows, (unit % chunks + 1) * kAttentionRows);
    for (auto row = unit % chunks * kAttentionRows; row < row_end; ++row) {
      const float* query = fused_qkv.data() + row * stride + head * head_width;
      const auto last_key = first_position + row;  // Causal: keys 0..last_key inclusive.
      for (std::size_t key = 0; key <= last_key; ++key) {
        scores[key] = dot(query, keys.data() + key * width + head * head_width, head_width) * scale;
      }
      ops::ref::softmax(std::span(scores).first(last_key + 1));
      float* out = output.data() + row * width + head * head_width;
      std::fill(out, out + head_width, 0.0F);
      for (std::size_t key = 0; key <= last_key; ++key) {
        axpy(out, values.data() + key * width + head * head_width, scores[key], head_width);
      }
    }
  }
}
}  // namespace nanoinfer::kernels
