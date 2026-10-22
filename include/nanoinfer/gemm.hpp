#pragma once

#include <cstddef>
#include <span>
#include <string_view>

namespace nanoinfer {
class ThreadPool;
}

namespace nanoinfer::kernels {

// The GEMM optimization steps, kept side by side so each one can be tested
// against the scalar reference and benchmarked on its own.
enum class GemmStep {
  naive,      // 1. i-j-k triple loop
  reordered,  // 2. i-k-j loop order, contiguous inner access
  blocked,    // 3. M/N/K cache blocking around the i-k-j loop
  packed,     // 4. packed A/B panels and a portable 6x16 micro-kernel
  avx2,       // 5. packed panels and an AVX2/FMA 6x16 register-blocked micro-kernel
};

[[nodiscard]] std::string_view name(GemmStep step) noexcept;

struct CpuFeatures {
  bool avx2 = false;
  bool fma = false;
  bool avx512f = false;
};

// Instruction support reported by the CPU and enabled by the operating system.
[[nodiscard]] const CpuFeatures& cpu_features() noexcept;

// True when this build contains the step and the running CPU can execute it.
[[nodiscard]] bool available(GemmStep step) noexcept;

// The fastest available step: AVX2 when supported, otherwise the packed
// portable kernel.
[[nodiscard]] GemmStep best_step() noexcept;

// C[M,N] = A[M,K] * B, where B is [K,N] row-major, or [N,K] row-major when
// transpose_b is set (the tied output embedding). C is overwritten. All
// buffers are contiguous and must not overlap.
//
// Step 6: with a pool, the packed and avx2 steps split B packing and the
// MC x (8 NR) output tiles across its threads. Every tile is still computed by
// one thread in a fixed order, so the result is bit-identical for any thread
// count. Steps 1-3 ignore the pool.
void gemm(GemmStep step, std::span<const float> a, std::span<const float> b, std::size_t rows,
          std::size_t columns, std::size_t inner, std::span<float> output, bool transpose_b = false,
          ThreadPool* pool = nullptr);

}  // namespace nanoinfer::kernels
