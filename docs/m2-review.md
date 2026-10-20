# M2 review packet

Review with `git diff main...m2`. The base is approved M1 commit
`3184edd0b73e87af11e1556072652979cafcb3b7`. M2 is submitted for review on its
own branch; M3 has not started.

## What was built

A single-threaded, float32 reference transformer with embedding lookup, naive
matrix multiplication, bias, residual addition, LayerNorm, tanh GELU, stable
softmax and causal multi-head attention. `gpt2ref` writes every prompt logit,
each complete block output, final LayerNorm output, and greedy continuation IDs.
Generation recomputes the whole prefix at each step.

Both M1 follow-ups are included: strict GPT-2 tensor schema checks against the
header configuration, and a sanitizer regression test operating on a temporary
project copy while checking that original result files remain unchanged.

## Acceptance contract

The owner explicitly approved hidden-state `atol=1e-4` plus `rtol=1e-6`.
The check is elementwise:
`abs(actual - expected) <= 1e-4 + 1e-6 * abs(expected)`.
It applies to all 12 block outputs and final LayerNorm. This is a clarification
for M2; the spec file has not been edited. The logits limit remains strictly
below `1e-3` maximum absolute difference. Every golden prompt must match all
32 greedy token IDs exactly.

The pretrained fixtures have large hidden activations whose float32 spacing
exceeds the original absolute bound. The optional
`scripts/check_hidden_precision.py` diagnosis is recorded separately in
`results/m2-hidden-precision-diagnostic.json`; it is not an acceptance test.
Runtime arithmetic remains float32.

| Acceptance criterion | Status and measured evidence |
|---|---|
| Every pretrained prompt's logits | Met in Debug and ASan; maximum absolute difference `0.00018310546875 < 0.001` |
| Every complete block state and final LayerNorm | Met under the approved combined bound; largest elementwise error/bound ratio `0.943782678711 <= 1` |
| Greedy generation | Met in Debug and ASan; all 32 IDs match exactly for each of the three prompts |
| Local tests | Met; CTest 7/7 in each preset, GoogleTest 20/20 at seeds 42 and 1234, Python 30/30 |
| CI tiny-model golden check | Met in Windows and Ubuntu Debug/ASan, including all 32 greedy IDs |
| M1 follow-ups | Met; schema rejection and temporary-copy sanitizer regressions pass |

Maximum absolute hidden error is `0.00048828125`. This exceeds the original
absolute-only limit; acceptance uses the explicitly approved elementwise bound.

## Evidence

The complete local output is committed, with machine-specific paths redacted:

| Check | Evidence |
|---|---|
| Debug configure/build | [configure](../results/m2-debug-configure.txt), [build](../results/m2-debug-build.txt) |
| Debug CTest, including tiny golden comparison | [full output](../results/m2-debug-tests.txt) |
| Debug seed 1234 | [full output](../results/m2-debug-seed1234.txt) |
| ASan configure/build | [configure](../results/m2-asan-configure.txt), [build](../results/m2-asan-build.txt) |
| ASan CTest, including tiny golden comparison | [full output](../results/m2-asan-tests.txt) |
| ASan seed 1234 | [full output](../results/m2-asan-seed1234.txt) |
| Python tests | [full output](../results/m2-pytest.txt) |
| Pretrained Debug | [full output](../results/m2-debug-pretrained.txt), [all numerical errors](../results/m2-debug-pretrained.json) |
| Pretrained ASan | [full output](../results/m2-asan-pretrained.txt), [all numerical errors](../results/m2-asan-pretrained.json) |
| Aggregated pretrained acceptance and actual greedy IDs | [summary](../results/m2-summary.json) |
| Sanitizer test isolation | [original failure](../results/m2-sanitizer-isolation-regression.txt), [passing regression](../results/m2-sanitizer-isolation.txt) |
| Tensor shape validation | [original failure](../results/m2-schema-regression.txt), [passing regression](../results/m2-schema-tests.txt) |
| GELU cubic rounding | [original failure](../results/m2-gelu-order-regression.txt), [passing operations](../results/m2-reference-ops-tests.txt) |

`scripts/test-m2.ps1 -Preset debug -Pretrained` and the corresponding `asan`
command reproduce the local suite. Run `python scripts/summarize_m2.py` after
both; it rejects incomplete generation, incorrect thresholds, failing errors,
missing layers or mismatching saved token IDs. Its summary records model,
golden-file and executable SHA-256 values. These are correctness runs, not
benchmarks. Each native process has one thread; separate golden cases may run
as independent processes.

The implementation at `1a9f8a5` passed all five jobs in
[run 39430228523](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39430228523):
Windows and Ubuntu Debug/ASan, plus lint. Ubuntu ASan includes UBSan with
`-fno-sanitize-recover=all` and the deliberate-overflow failure probe.
The delivery message identifies the final commit and its gating run after the
evidence and documentation commit. The earlier implementation run above is
not a substitute for that final gate. TSan is skipped because M2 has no runtime
worker threads.

One initial local ASan invocation stalled in the malformed-shape subprocess
without a sanitizer diagnostic. A fresh invocation rejected that same file
correctly. The stalled run was stopped and the complete suite rerun with
`ctest --preset asan --verbose --no-tests=error --timeout 180`; all seven checks
passed. The committed ASan output is that completed rerun. Its pretrained
generation check and alternate-seed unit run also completed successfully.

## Files changed

| Area | Files |
|---|---|
| Reference operations | `include/nanoinfer/ref_ops.hpp`, `src/ops/ref/ops.cpp`, `tests/test_ref_ops.cpp` |
| Forward and greedy generation | `include/nanoinfer/reference.hpp`, `src/runtime/reference.cpp` |
| Correctness CLI | `src/tools/reference.hpp`, `src/tools/reference.cpp`, `src/platform/reference_main_windows.cpp`, `src/platform/reference_main_posix.cpp` |
| Loader schema and inspector | `include/nanoinfer/model.hpp`, `src/model/loader.cpp`, `src/tools/inspect.hpp`, `src/tools/inspect.cpp`, both `src/platform/inspect_main_*.cpp` |
| Schema and format tests | `tests/model_schema.py`, `tests/test_loader.cpp`, `tests/loader_integration.py` |
| Forward comparison and boundary tests | `tests/golden_forward.py`, `tests/test_golden_comparison.py`, `tests/reference_contract.py` |
| Test integration and reproduction | `CMakeLists.txt`, `tests/test_sanitize_results.ps1`, `scripts/test-m2.ps1`, `scripts/summarize_m2.py`, `scripts/check_hidden_precision.py` |
| Documentation and evidence | `README.md`, `docs/gpt2-contract.md`, `golden/README.md`, this packet, `results/m2-*` |

## GPT-2 details and scope

- Conv1D weights retain `[in, out]` layout; output projection uses `wte` transposed.
- LayerNorm uses population variance and epsilon `1e-5`.
- GELU uses the tanh formula, rounding the cubic term before its coefficient.
- Causal attention includes position `i`, with separate head indexing and
  scaling by the square root of head width.
- Full-prefix recomputation uses positions `0..n-1`, preserving absolute
  positions as generation grows.
- Greedy ties choose the lowest ID; there is no EOS early stop.
- Context-limit, invalid-ID, Unicode-path, full-context and prefix-invariance
  checks run in CTest. Random operation tests accept and print `--seed=N`.
- CI generates a seeded tiny model and compares its native forward outputs and
  32 greedy tokens with PyTorch. Pretrained GPT-2 validation is local.
- No optimized kernels exist yet, so no optimized/reference differential pair
  or GEMM performance claim is made. KV cache, sampling, threads, int8,
  roofline analysis and performance benchmarks remain later milestones.
