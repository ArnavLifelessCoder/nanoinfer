#pragma once

#include <cstddef>
#include <cstdint>

namespace nanoinfer::kernels::detail {
// AVX2/FMA vector primitives for the decode GEMV, defined in gemv_avx2.cpp.
// Call only after a runtime CPU check.
void axpy_avx2(float* y, const float* source, float x, std::size_t count);
float dot_avx2(const float* a, const float* b, std::size_t count);
// y += x * float(q), dequantizing int8 in the inner loop.
void axpy_q8_avx2(float* y, const std::int8_t* q, float x, std::size_t count);
// sum over groups g of scales[g] * (x . float(q)) for count values in groups
// of `group`; group must be a multiple of 8.
float dot_q8_avx2(const float* x, const std::int8_t* q, const float* scales, std::size_t count,
                  std::size_t group);
}  // namespace nanoinfer::kernels::detail
