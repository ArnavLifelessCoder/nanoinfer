// Compiled with AVX2/FMA. No standard library templates (see gemm_avx2.cpp).
#include <immintrin.h>

#include <cstddef>

namespace nanoinfer::bench {
// Twelve independent FMA chains keep both FMA pipes busy despite the
// four-cycle FMA latency: 2 pipes x 4 cycles = 8 chains needed at minimum.
float fma_throughput(std::size_t iterations) {
  const __m256 x = _mm256_set1_ps(0.999999F), y = _mm256_set1_ps(1e-7F);
  __m256 a0 = _mm256_set1_ps(1), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0;
  __m256 a6 = a0, a7 = a0, a8 = a0, a9 = a0, a10 = a0, a11 = a0;
  for (std::size_t i = 0; i < iterations; ++i) {
    a0 = _mm256_fmadd_ps(a0, x, y);
    a1 = _mm256_fmadd_ps(a1, x, y);
    a2 = _mm256_fmadd_ps(a2, x, y);
    a3 = _mm256_fmadd_ps(a3, x, y);
    a4 = _mm256_fmadd_ps(a4, x, y);
    a5 = _mm256_fmadd_ps(a5, x, y);
    a6 = _mm256_fmadd_ps(a6, x, y);
    a7 = _mm256_fmadd_ps(a7, x, y);
    a8 = _mm256_fmadd_ps(a8, x, y);
    a9 = _mm256_fmadd_ps(a9, x, y);
    a10 = _mm256_fmadd_ps(a10, x, y);
    a11 = _mm256_fmadd_ps(a11, x, y);
  }
  __m256 sum = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)),
                             _mm256_add_ps(_mm256_add_ps(a4, a5), _mm256_add_ps(a6, a7)));
  sum = _mm256_add_ps(sum, _mm256_add_ps(_mm256_add_ps(a8, a9), _mm256_add_ps(a10, a11)));
  return _mm256_cvtss_f32(sum);
}

// One dependent FMA chain: each step waits for the previous result, so the
// loop runs at one FMA per FMA-latency cycles. Timing it gives the core clock.
float fma_chain(std::size_t iterations) {
  const __m256 x = _mm256_set1_ps(0.999999F), y = _mm256_set1_ps(1e-7F);
  __m256 a = _mm256_set1_ps(1);
  for (std::size_t i = 0; i < iterations; ++i) a = _mm256_fmadd_ps(a, x, y);
  return _mm256_cvtss_f32(a);
}
}  // namespace nanoinfer::bench
