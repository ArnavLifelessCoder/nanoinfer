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

void axpy_q8_scalar(float* y, const std::int8_t* q, float x, std::size_t count) {
  for (std::size_t j = 0; j < count; ++j) y[j] += x * static_cast<float>(q[j]);
}

float dot_q8_scalar(const float* x, const std::int8_t* q, const float* scales, std::size_t count,
                    std::size_t group) {
  float total = 0.0F;
  for (std::size_t start = 0, g = 0; start < count; start += group, ++g) {
    float sum = 0.0F;
    for (auto i = start; i < start + group; ++i) sum += x[i] * static_cast<float>(q[i]);
    total += sum * scales[g];
  }
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
void validate(const QuantizedMatrix& b, std::size_t columns, std::size_t inner) {
  if (b.group == 0 || inner % b.group != 0 ||
      b.values.size() != ops::ref::elements(inner, columns) ||
      b.scales.size() != ops::ref::elements(inner / b.group, columns)) {
    throw std::invalid_argument("q8 matrix shape mismatch");
  }
}

void gemv(GemvKernel kernel, std::span<const float> x, const QuantizedMatrix& b,
          std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b,
          ThreadPool* pool) {
  if (inner == 0 || columns == 0 || x.size() != inner || output.size() != columns) {
    throw std::invalid_argument("gemv shape mismatch");
  }
  validate(b, columns, inner);
  if (!available(kernel)) throw std::runtime_error("the AVX2 GEMV needs a CPU with AVX2 and FMA");
  const auto group = b.group;
#if defined(NANOINFER_HAVE_AVX2_KERNEL)
  // The vector dot product handles groups of whole 8-float vectors.
  const bool vector = kernel == GemvKernel::avx2;
  const auto axpy = vector ? detail::axpy_q8_avx2 : axpy_q8_scalar;
  const auto dot = vector && group % 8 == 0 ? detail::dot_q8_avx2 : dot_q8_scalar;
#else
  const auto axpy = axpy_q8_scalar;
  const auto dot = dot_q8_scalar;
#endif
  const std::int8_t* q = b.values.data();
  const float* scales = b.scales.data();
  const float* input = x.data();
  float* y = output.data();
  if (transpose_b) {
    const auto groups = inner / group;
    run(pool, (columns + kDotRows - 1) / kDotRows, [&](std::size_t begin, std::size_t end) {
      for (std::size_t unit = begin; unit < end; ++unit) {
        const auto last = std::min(columns, (unit + 1) * kDotRows);
        for (auto row = unit * kDotRows; row < last; ++row) {
          y[row] = dot(input, q + row * inner, scales + row * groups, inner, group);
        }
      }
    });
    return;
  }
  // One partial row per quantization group: sum x_k * q[k, :] over the group's
  // rows, then apply that group's scales once. Then add the partial rows in
  // group order, so the result is independent of the thread count.
  const auto blocks = inner / group;
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
      for (auto k = block * group; k < (block + 1) * group; ++k) {
        axpy(row, q + k * columns + j0, input[k], width);
      }
      const float* scale = scales + block * columns + j0;
      for (std::size_t j = 0; j < width; ++j) row[j] *= scale[j];
    }
  });
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
