// Decode GEMV: fixed-order blocked matrix-vector products, optionally split
// across a thread pool. Built with the baseline ISA; AVX2 primitives come from
// gemv_avx2.cpp after a runtime check.
#include "nanoinfer/gemv.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "kernels/vector_kernels.hpp"
#include "nanoinfer/gemm.hpp"
#include "nanoinfer/ref_ops.hpp"
#include "nanoinfer/thread_pool.hpp"

namespace nanoinfer::kernels {
namespace {
// 32 rows of B per partial sum; 1024 output columns per work unit. Both are
// shape constants, never derived from the thread count.
constexpr std::size_t kRowBlock = 32;
constexpr std::size_t kColumnChunk = 1024;
// Rows per work unit of the transposed product.
constexpr std::size_t kDotRows = 64;

void axpy_scalar(float* y, const float* source, float x, std::size_t count) {
  for (std::size_t j = 0; j < count; ++j) y[j] += x * source[j];
}

float dot_scalar(const float* a, const float* b, std::size_t count) {
  float total = 0.0F;
  for (std::size_t i = 0; i < count; ++i) total += a[i] * b[i];
  return total;
}

template <typename Fn>
void run(ThreadPool* pool, std::size_t count, const Fn& fn) {
  if (pool == nullptr) {
    if (count != 0) fn(std::size_t{0}, count);
  } else {
    pool->parallel_for(count, fn);
  }
}
}  // namespace

bool available(GemvKernel kernel) noexcept {
  return kernel == GemvKernel::scalar || available(GemmStep::avx2);
}

GemvKernel best_gemv() noexcept {
  return available(GemvKernel::avx2) ? GemvKernel::avx2 : GemvKernel::scalar;
}

void gemv(GemvKernel kernel, std::span<const float> x, std::span<const float> b,
          std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b,
          ThreadPool* pool) {
  if (inner == 0 || columns == 0 || x.size() != inner ||
      b.size() != ops::ref::elements(inner, columns) || output.size() != columns) {
    throw std::invalid_argument("gemv shape mismatch");
  }
  if (!available(kernel)) throw std::runtime_error("the AVX2 GEMV needs a CPU with AVX2 and FMA");
#if defined(NANOINFER_HAVE_AVX2_KERNEL)
  const bool vector = kernel == GemvKernel::avx2;
  const auto axpy = vector ? detail::axpy_avx2 : axpy_scalar;
  const auto dot = vector ? detail::dot_avx2 : dot_scalar;
#else
  const auto axpy = axpy_scalar;
  const auto dot = dot_scalar;
#endif
  const float* weights = b.data();
  const float* input = x.data();
  float* y = output.data();
  if (transpose_b) {
    // y[v] = x . B[v, :]: every row of B is read once, contiguously.
    run(pool, (columns + kDotRows - 1) / kDotRows, [&](std::size_t begin, std::size_t end) {
      for (std::size_t unit = begin; unit < end; ++unit) {
        const auto last = std::min(columns, (unit + 1) * kDotRows);
        for (auto row = unit * kDotRows; row < last; ++row) {
          y[row] = dot(input, weights + row * inner, inner);
        }
      }
    });
    return;
  }
  const auto blocks = (inner + kRowBlock - 1) / kRowBlock;
  const auto chunks = (columns + kColumnChunk - 1) / kColumnChunk;
  thread_local std::vector<float> partial;
  partial.assign(blocks * columns, 0.0F);
  float* sums = partial.data();
  run(pool, blocks * chunks, [&](std::size_t begin, std::size_t end) {
    for (std::size_t unit = begin; unit < end; ++unit) {
      const auto block = unit / chunks, chunk = unit % chunks;
      const auto j0 = chunk * kColumnChunk;
      const auto width = std::min(columns, j0 + kColumnChunk) - j0;
      float* row = sums + block * columns + j0;
      const auto k_end = std::min(inner, (block + 1) * kRowBlock);
      for (auto k = block * kRowBlock; k < k_end; ++k) {
        axpy(row, weights + k * columns + j0, input[k], width);
      }
    }
  });
  // Add the partial rows in block order, so the order never depends on threads.
  run(pool, chunks, [&](std::size_t begin, std::size_t end) {
    for (std::size_t chunk = begin; chunk < end; ++chunk) {
      const auto j0 = chunk * kColumnChunk;
      const auto j1 = std::min(columns, j0 + kColumnChunk);
      for (auto j = j0; j < j1; ++j) {
        float total = sums[j];
        for (std::size_t block = 1; block < blocks; ++block) total += sums[block * columns + j];
        y[j] = total;
      }
    }
  });
}
}  // namespace nanoinfer::kernels
