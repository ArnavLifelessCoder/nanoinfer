// Compiled with AVX2/FMA. No standard library templates (see gemm_avx2.cpp).
#include <immintrin.h>

#include <cstddef>

#include "kernels/vector_kernels.hpp"

namespace nanoinfer::kernels::detail {
void axpy_avx2(float* y, const float* source, float x, std::size_t count) {
  const __m256 scale = _mm256_set1_ps(x);
  std::size_t j = 0;
  for (; j + 32 <= count; j += 32) {
    _mm256_storeu_ps(y + j,
                     _mm256_fmadd_ps(scale, _mm256_loadu_ps(source + j), _mm256_loadu_ps(y + j)));
    _mm256_storeu_ps(y + j + 8, _mm256_fmadd_ps(scale, _mm256_loadu_ps(source + j + 8),
                                                _mm256_loadu_ps(y + j + 8)));
    _mm256_storeu_ps(y + j + 16, _mm256_fmadd_ps(scale, _mm256_loadu_ps(source + j + 16),
                                                 _mm256_loadu_ps(y + j + 16)));
    _mm256_storeu_ps(y + j + 24, _mm256_fmadd_ps(scale, _mm256_loadu_ps(source + j + 24),
                                                 _mm256_loadu_ps(y + j + 24)));
  }
  for (; j + 8 <= count; j += 8) {
    _mm256_storeu_ps(y + j,
                     _mm256_fmadd_ps(scale, _mm256_loadu_ps(source + j), _mm256_loadu_ps(y + j)));
  }
  for (; j < count; ++j) y[j] += x * source[j];  // Scalar tail: no load past the end.
}

float dot_avx2(const float* a, const float* b, std::size_t count) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 32 <= count; i += 32) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
    s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
  }
  for (; i + 8 <= count; i += 8) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  }
  const __m256 sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, sum);
  float total = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) +
                ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
  for (; i < count; ++i) total += a[i] * b[i];  // Scalar tail: no load past the end.
  return total;
}
}  // namespace nanoinfer::kernels::detail
