# M4 review packet

Review `git diff m3...m4`. M4 adds GEMM optimization steps 1-5 with runtime
CPU dispatch and a scalar fallback; multithreading (step 6) is M5.

| Acceptance criterion | Evidence | Status |
|---|---|---|
| Differential tests vs the reference GEMM, random and awkward shapes, relative error under 1e-5 | `tests/test_gemm.cpp`: every step, both B layouts, M/N/K in {1, 7, 63, 65, 767, 769}, register-tile edges and KC/MC/NC block crossings, plus random shapes; error per output relative to `sum_p abs(a_ip * b_pj)` and relative Frobenius error both below 1e-5; seeds 42 and 1234; Debug and ASan | Met |
| Benchmark table for each step committed | [m4-gemm.md](../results/m4-gemm.md), [m4-gemm.json](../results/m4-gemm.json), produced by `bench/run_gemm.py` | Met |
| Forward pass still matches golden outputs | `golden_optimized` (CTest, tiny model, every step) and [pretrained report](../results/m4-pretrained-optimized.json): logits within 1e-3 of PyTorch, 32/32 greedy tokens, hidden states per the rule below, and the same checks against the scalar reference forward | Met |

Design, tail handling and dispatch are described in [kernels.md](kernels.md).

## Hidden-state rule for optimized kernels (owner decision)

Steps 1-3 sum each output in the same order as the scalar reference and are
bit-identical to it. Steps 4-5 split K into 256-wide blocks. On the pretrained
model that moved layer 11 outside the M2 fp32 bound (up to ten times the
allowed error on the ~3000-magnitude residual channel), while logits and
greedy tokens still matched. Measured against a float64 PyTorch run, the
blocked kernels were 3 to 25 times closer to the exact result than both
PyTorch float32 and the scalar reference: the bound penalized a different,
more accurate summation order.

The owner chose to keep the PyTorch float32 checks for logits and tokens and
to judge hidden states also against float64: `python/oracle.py` writes
`golden/oracle_prompt_N.npz` (float64 block outputs and final LayerNorm). A
hidden tensor passes if it meets the approved fp32 bound, or if its maximum
error against float64 is no larger than PyTorch float32's own error plus the
approved 1e-4. The same rule, with the reference's float64 error, applies to the
comparison with the scalar reference forward. Each tensor's three float64
errors are recorded in the report. CTest builds a tiny-model oracle
(`oracle_tiny`) so the rule runs in CI.

## Numbers

All numbers come from `results/`. Single thread pinned to one core, seven
repetitions; peak uses the clock measured under FMA load (1.95 GHz under the
Windows Balanced plan, which was not changed). The AVX2 step reaches 85.4% and
82.9% of single-core peak on the two shapes; the table explains why steps 3 and
4 do not improve on step 2 with this compiler.

## Reproduce

```powershell
cmake --preset release -DNANOINFER_BENCHMARKS=ON
cmake --build --preset release
python bench/run_gemm.py --build build/release
python -m python.oracle --golden golden
python -m tests.golden_optimized --reference-executable build/release/gpt2ref.exe `
  --executable build/release/gpt2opt.exe --model models/gpt2.bin --golden golden `
  --directory build/release/optimized-pretrained --report results/m4-pretrained-optimized.json `
  --kernels naive reordered blocked packed avx2 --timeout 3600
```
