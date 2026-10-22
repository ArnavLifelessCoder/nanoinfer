#pragma once

#include <cstddef>

namespace nanoinfer::kernels::detail {
// AVX2/FMA vector primitives for the decode GEMV, defined in gemv_avx2.cpp.
// Call only after a runtime CPU check.
void axpy_avx2(float* y, const float* source, float x, std::size_t count);
float dot_avx2(const float* a, const float* b, std::size_t count);
}  // namespace nanoinfer::kernels::detail
