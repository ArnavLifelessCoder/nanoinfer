// Compiled with AVX2/FMA code generation. Keep this file free of standard
// library templates and other inline functions shared with portable
// translation units: the linker may keep any one copy of such a function, and
// an AVX2-encoded copy would fault on a CPU without AVX2.
#include <immintrin.h>

#include <cstddef>

#include "kernels/micro_kernel.hpp"

namespace nanoinfer::kernels::detail {
namespace {
inline void store_row(float* row, __m256 low, __m256 high, bool accumulate) {
  if (accumulate) {
    low = _mm256_add_ps(_mm256_loadu_ps(row), low);
    high = _mm256_add_ps(_mm256_loadu_ps(row + 8), high);
  }
  _mm256_storeu_ps(row, low);
  _mm256_storeu_ps(row + 8, high);
}
}  // namespace

void micro_kernel_avx2(std::size_t depth, const float* a, const float* b, float* c, std::size_t ldc,
                       std::size_t rows, std::size_t cols, bool accumulate) {
  static_assert(kMr == 6 && kNr == 16, "the register tile is written out for 6x16");
  __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
  __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
  __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
  __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
  __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
  __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();
  for (std::size_t p = 0; p < depth; ++p) {
    // Packed B rows are 64-byte aligned: the panel base is, and each row is 16 floats.
    const __m256 b0 = _mm256_load_ps(b);
    const __m256 b1 = _mm256_load_ps(b + 8);
    __m256 x = _mm256_broadcast_ss(a);
    c00 = _mm256_fmadd_ps(x, b0, c00);
    c01 = _mm256_fmadd_ps(x, b1, c01);
    x = _mm256_broadcast_ss(a + 1);
    c10 = _mm256_fmadd_ps(x, b0, c10);
    c11 = _mm256_fmadd_ps(x, b1, c11);
    x = _mm256_broadcast_ss(a + 2);
    c20 = _mm256_fmadd_ps(x, b0, c20);
    c21 = _mm256_fmadd_ps(x, b1, c21);
    x = _mm256_broadcast_ss(a + 3);
    c30 = _mm256_fmadd_ps(x, b0, c30);
    c31 = _mm256_fmadd_ps(x, b1, c31);
    x = _mm256_broadcast_ss(a + 4);
    c40 = _mm256_fmadd_ps(x, b0, c40);
    c41 = _mm256_fmadd_ps(x, b1, c41);
    x = _mm256_broadcast_ss(a + 5);
    c50 = _mm256_fmadd_ps(x, b0, c50);
    c51 = _mm256_fmadd_ps(x, b1, c51);
    a += kMr;
    b += kNr;
  }
  if (rows == kMr && cols == kNr) {
    store_row(c, c00, c01, accumulate);
    store_row(c + ldc, c10, c11, accumulate);
    store_row(c + 2 * ldc, c20, c21, accumulate);
    store_row(c + 3 * ldc, c30, c31, accumulate);
    store_row(c + 4 * ldc, c40, c41, accumulate);
    store_row(c + 5 * ldc, c50, c51, accumulate);
    return;
  }
  // Edge tile: spill to a local buffer and store only the valid rows and
  // columns, so no vector access touches memory past the end of C.
  alignas(32) float tile[kMr * kNr];
  _mm256_store_ps(tile, c00);
  _mm256_store_ps(tile + 8, c01);
  _mm256_store_ps(tile + 16, c10);
  _mm256_store_ps(tile + 24, c11);
  _mm256_store_ps(tile + 32, c20);
  _mm256_store_ps(tile + 40, c21);
  _mm256_store_ps(tile + 48, c30);
  _mm256_store_ps(tile + 56, c31);
  _mm256_store_ps(tile + 64, c40);
  _mm256_store_ps(tile + 72, c41);
  _mm256_store_ps(tile + 80, c50);
  _mm256_store_ps(tile + 88, c51);
  for (std::size_t i = 0; i < rows; ++i) {
    float* row = c + i * ldc;
    const float* source = tile + i * kNr;
    for (std::size_t j = 0; j < cols; ++j) {
      row[j] = accumulate ? row[j] + source[j] : source[j];
    }
  }
}
}  // namespace nanoinfer::kernels::detail
