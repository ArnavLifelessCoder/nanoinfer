#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace nanoinfer::kernels {

// An int8 weight matrix with one f32 scale per `group` consecutive values
// along the inner (K) dimension: B is [K, N] with scales [K / group, N], or,
// for a transposed product, [N, K] with scales [N, K / group]. The value of an
// element is float(q) * scale, exactly as python/quantize.py dequantizes.
struct QuantizedMatrix {
  std::span<const std::int8_t> values;
  std::span<const float> scales;
  std::size_t group = 0;
};

// Throws unless the spans match a K x N (or N x K) matrix grouped along K.
void validate(const QuantizedMatrix& b, std::size_t columns, std::size_t inner);

}  // namespace nanoinfer::kernels
