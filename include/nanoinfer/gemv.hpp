#pragma once

#include <cstddef>
#include <span>

#include "nanoinfer/quantized.hpp"

namespace nanoinfer {
class ThreadPool;
}

namespace nanoinfer::kernels {

// Matrix-vector product for single-token decode, which is limited by memory
// bandwidth: the weights are streamed once, sequentially.
enum class GemvKernel { scalar, avx2 };

[[nodiscard]] bool available(GemvKernel kernel) noexcept;
[[nodiscard]] GemvKernel best_gemv() noexcept;

// y[N] = x[K] * B, where B is [K,N] row-major, or [N,K] when transpose_b.
// The non-transposed product sums fixed 32-row blocks of B into partial rows
// and then adds the partial rows in block order; the transposed product is one
// dot product per output. Both orders depend only on the shapes, so the result
// is bit-identical for any thread count. pool may be null (one thread).
void gemv(GemvKernel kernel, std::span<const float> x, std::span<const float> b,
          std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b = false,
          ThreadPool* pool = nullptr);

// int8 weights (see QuantizedMatrix), dequantized in the inner loop: int8 is
// widened to float, multiplied by x, and each group's scale is applied once.
// About 1.125 bytes are read per weight instead of 4.
void gemv(GemvKernel kernel, std::span<const float> x, const QuantizedMatrix& b,
          std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b = false,
          ThreadPool* pool = nullptr);

}  // namespace nanoinfer::kernels
