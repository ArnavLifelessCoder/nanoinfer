# GEMM kernels

`include/nanoinfer/gemm.hpp` exposes the five single-thread optimization steps
side by side, so each one can be tested against the scalar reference
(`ops::ref::matmul`) and benchmarked on its own. All of them compute
`C[M,N] = A[M,K] * B`, where `B` is the Conv1D `[in, out]` layout, or the
`[N, K]` tied embedding when `transpose_b` is set (the logits product).

| Step | Idea | Source |
|---|---|---|
| 1. naive | i-j-k triple loop, one dot product per output | `src/kernels/gemm.cpp` |
| 2. reordered | i-k-j: the inner loop streams a row of B into a row of C, so it is contiguous and vectorizable | same |
| 3. blocked | i-k-j inside 64 x 256 x 256 tiles of C/B, for cache reuse | same |
| 4. packed | GotoBLAS loop nest; A and B are copied into zero-padded MR x KC and KC x NR panels, then a portable 6 x 16 micro-kernel | same |
| 5. avx2 | the same packed loop nest with a 6 x 16 AVX2/FMA register tile (12 YMM accumulators) | `src/kernels/gemm_avx2.cpp` |

Blocking constants live in `src/kernels/micro_kernel.hpp`: MR = 6, NR = 16,
KC = 256, MC = 144, NC = 3072. A KC x NR panel of B (16 KiB) and an MR x KC
panel of A (6 KiB) fit in the 32 KiB L1D; an MC x KC block of A (144 KiB) fits
in L2; a KC x NC block of B (3 MiB) fits in L3.

## Tails and bounds

Packing pads partial panels with zeros, so the micro-kernel always runs a full
6 x 16 tile over full panels and never reads outside a packed buffer. Packed
buffers are 64-byte aligned and every packed B row is 16 floats, so the
aligned `_mm256_load_ps` loads are always in bounds and aligned. Only the valid
rows and columns of a tile are stored: an edge tile is spilled to a local
buffer and copied element by element, so no vector store touches memory past
the end of `C`. The differential tests run every step on M, N and K values of
1, 7, 63, 65, 767 and 769, register-tile edges (5 x 15, 6 x 16) and shapes that
cross the KC, MC and NC blocks, under ASan and UBSan.

## Dispatch and the scalar fallback

`gemm_avx2.cpp` is the only translation unit built with AVX2/FMA code
generation (`/arch:AVX2`, or `-mavx2 -mfma`). It contains no standard library
templates or other inline functions shared with portable code: a linker keeps
one copy of such a function, and an AVX2-encoded copy would fault on a CPU
without AVX2. `kernels::cpu_features()` checks CPU support and operating system
state (`__builtin_cpu_supports` on GCC and Clang; `CPUID` plus `XGETBV` on
MSVC). `best_step()` returns `avx2` when AVX2 and FMA are usable and the
portable packed step otherwise; asking for `avx2` on a CPU without it throws.
AVX-512F is detected and reported but has no kernel: on Zen 4, 512-bit FMAs
issue as two 256-bit halves, so the AVX2 tile already reaches the same peak.

The benchmark-only `release` preset compiles everything with
`/arch:AVX2` (MSVC) or `-march=native`; steps 2 and 3 then rely on compiler
auto-vectorization, through one shared contiguous `axpy` loop.

## Results

`bench/run_gemm.py` writes [m4-gemm.json](../results/m4-gemm.json) and
[m4-gemm.md](../results/m4-gemm.md): one pinned thread, seven repetitions,
median and range. Peak is `cores x clock x FMA units x lanes x 2` with one core,
Zen 4's two FMA pipes and eight fp32 lanes. The clock is measured under FMA load
from a dependent FMA chain (four-cycle latency); the independent FMA probe in
the same JSON reaches that peak, which checks the factors.

Reading the table:

- Step 1 falls to 0.2 GFLOPS on the 1024 x 3072 x 768 shape: its inner loop walks
  a column of B with a 12 KiB stride, so nearly every load misses.
- Step 2 is the large jump. The contiguous inner loop vectorizes and streams B.
- Step 3 adds little at these sizes. A row of B or C is at most 12 KiB, so the
  i-k-j order already reuses cache well; blocking helps more once rows outgrow L1/L2.
- Step 4 is slower than steps 2 and 3. Packing alone does not speed anything up:
  MSVC does not vectorize the portable 6 x 16 tile, so it runs scalar
  arithmetic. Packing exists to feed a register-blocked micro-kernel; step 5 is
  that kernel. Step 4 remains the portable fallback for CPUs without AVX2.
- Step 5 keeps the whole tile in registers and issues two FMAs per loaded B
  vector pair per row, reaching over 80% of single-core peak.

The machine was on AC power under Windows' Balanced plan, which held the
measured core clock near 1.95 GHz under FMA load, well below the CPU's rated
boost. Power settings were not changed; the peak uses the measured clock, so
the percentages are consistent with the clock the kernels actually ran at.
Multithreading (step 6) and the decode GEMV are M5 work.
