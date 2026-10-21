#pragma once

#include <cstddef>

namespace nanoinfer::kernels::detail {

// Register tile and cache blocks. A 6x16 float tile uses 12 YMM accumulators.
// KC x NR of B (16 KiB) plus MR x KC of A (6 KiB) fit in a 32 KiB L1D; an
// MC x KC block of A (144 KiB) fits in L2; a KC x NC block of B (3 MiB) fits in L3.
inline constexpr std::size_t kMr = 6;
inline constexpr std::size_t kNr = 16;
inline constexpr std::size_t kKc = 256;
inline constexpr std::size_t kMc = 144;
inline constexpr std::size_t kNc = 3072;

// Multiplies a packed MR x depth A panel by a packed depth x NR B panel. The
// rows x cols top-left part of the tile is written to c (row stride ldc),
// replacing it or adding to it. Packed panels are zero padded, so the kernel
// never reads outside them; only the valid part of the tile is stored.
using MicroKernel = void (*)(std::size_t depth, const float* a, const float* b, float* c,
                             std::size_t ldc, std::size_t rows, std::size_t cols, bool accumulate);

// Defined in gemm_avx2.cpp, which is the only translation unit compiled with
// AVX2/FMA code generation. Call it only after a runtime CPU check.
void micro_kernel_avx2(std::size_t depth, const float* a, const float* b, float* c, std::size_t ldc,
                       std::size_t rows, std::size_t cols, bool accumulate);

}  // namespace nanoinfer::kernels::detail
