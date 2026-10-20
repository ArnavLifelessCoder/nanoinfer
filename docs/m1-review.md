# M1 review packet

Date: 2026-10-20. Status: local M1 acceptance criteria met; awaiting owner review
and approval to push. M2 has not started.

## What was built

- C++20 CMake/Ninja skeleton with Debug, Release, ASan and Linux TSan presets.
- F32 GPT-2 exporter, versioned aligned binary format, printed JSON manifest and
  a small pretrained golden set. Conv1D orientation and tied embeddings are tested.
- Read-only Windows and POSIX mapping implementations under `src/platform/`.
- Validated C++ loader with immutable, zero-copy tensor views, shape and stride
  metadata, checked arithmetic, canonical layout checks and CRC32 verification.
- Native `gpt2inspect` command and Python round-trip checks for every tensor.
- Windows/Ubuntu Debug and ASan CI, formatting and Python lint checks, and a
  disabled Ubuntu TSan job to enable when runtime worker threads arrive.

The native executable imports only Windows and Microsoft C/C++ runtime DLLs;
see [the dependency inspection](../results/m1-runtime-dependencies.txt).

## Acceptance criteria

| Criterion | Result | Evidence |
| --- | --- | --- |
| Export real GPT-2, load it in C++, compare every tensor's checksum and shape with exporter output | PASS in Debug and ASan, all 148 tensors | Full pretrained logs below; [printed manifest](../results/m1-export-manifest.json) |
| Truncated or corrupted files give a clean error | PASS | Every truncation boundary and byte corruption of the C++ fixture, invalid metadata with recomputed checksums, overflow/overlap cases, and native CLI integration checks |
| Debug tests pass without compiler warnings | PASS | 12 C++ unit tests and all 4 CTest entries; complete log below |
| MSVC AddressSanitizer tests pass without findings or compiler warnings | PASS | Same tests and full pretrained round-trip; complete log below |
| Exporter and correctness tools pass their tests | PASS | 20 pytest tests; complete log below |
| Small golden set is saved, weights and specs are excluded | PASS | 3 fixtures, 997,382 compressed bytes; [artifact audit](../results/m1-summary.json) |
| Formatting and lint | PASS | [Full lint output](../results/m1-lint.txt) |
| GitHub CI execution | Pending approved push | No remote CI run is claimed |
| TSan | Not applicable to M1 | No runtime worker threads exist |

The golden set contains prompt token IDs, logits at every position, every complete
block output, final LayerNorm output and exactly 32 greedy continuation IDs.
Numerical comparison with a C++ forward pass is M2 work.

## Validation environment and reproduction

[The environment record](../results/m1-environment.json) identifies the machine,
OS, compiler, flags and Python libraries. Native validation used MSVC 19.44,
CMake/Ninja, Python 3.13, Transformers 4.57.1 and PyTorch 2.11.0+cu128 running
entirely on CPU with one thread.

The project venv reuses the existing PyTorch installation through system site
packages because wheel downloads stalled. Other pinned tools are installed in
the venv. Fresh installs and CI resolve the CPU wheel. This distinction is
recorded rather than treating the local wheel as a CPU-only distribution.
Windows uses Hugging Face's supported cache-copy fallback; the development
shell disables only the corresponding symlink-privilege advisory.

From PowerShell at the repository root:

```powershell
. .\scripts\dev-shell.ps1
python -m pytest --basetemp=build/pytest -v
python -m python.export --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --output models/gpt2.bin --golden golden
.\scripts\test.ps1 debug
.\scripts\test.ps1 asan
python scripts/verify_m1_artifacts.py
```

The second pinned-revision export reproduced the model SHA256 and every golden
array exactly, with empty stderr:
[reproducibility output](../results/m1-export-reproducibility.txt).

Full configure and build logs are also saved:
[Debug configure](../results/m1-debug-configure.txt),
[Debug build](../results/m1-debug-build.txt),
[ASan configure](../results/m1-asan-configure.txt),
[ASan build](../results/m1-asan-build.txt).

## Regression checks and limits

The empty-test-suite guard first reproduced a CMake registration-order error,
then remained in the local and CI commands after the fix:
[failure log](../results/m1-test-discovery-regression.txt).

The first ASan link exposed mismatched STL annotations in GoogleTest. The failing
link was saved before instrumenting dependency objects consistently:
[failure log](../results/m1-asan-link-regression.txt).

Earlier files containing `bootstrap` in their name record preliminary checks
with installed tooling. The final acceptance evidence is the Debug, ASan and
Python output included below.

The POSIX implementation is present but has not been executed locally. GitHub
Windows and Ubuntu validation awaits the first approved push. No performance
measurements, optimized kernels, quantization or runtime threading are claimed
in M1.

All work is in the folder containing the spec. The specs are unchanged and
ignored. Weights, build products, caches and the venv are ignored. Commits are
local on `main`, using the existing Git identity, and the remote is
[ArnavLifelessCoder/nanoinfer](https://github.com/ArnavLifelessCoder/nanoinfer).

## Files changed

The complete inventory and unabridged final test outputs follow. Only line endings
and trailing whitespace are normalized for storage.

```text
.clang-format
.clang-tidy
.gitattributes
.github/workflows/ci.yml
.gitignore
CMakeLists.txt
CMakePresets.json
README.md
docs/format.md
docs/gpt2-contract.md
docs/m1-review.md
golden/README.md
golden/manifest.json
golden/prompt_0.npz
golden/prompt_1.npz
golden/prompt_2.npz
include/nanoinfer/mapped_file.hpp
include/nanoinfer/model.hpp
pyproject.toml
python/__init__.py
python/export.py
python/model_format.py
results/m1-artifact-audit.txt
results/m1-asan-build.txt
results/m1-asan-configure.txt
results/m1-asan-gpt2.txt
results/m1-asan-link-regression.txt
results/m1-asan-tests.txt
results/m1-debug-build.txt
results/m1-debug-configure.txt
results/m1-debug-gpt2.txt
results/m1-debug-tests.txt
results/m1-environment.json
results/m1-export-bootstrap.txt
results/m1-export-manifest.json
results/m1-export-reproducibility.txt
results/m1-export-stderr.txt
results/m1-format-bootstrap.txt
results/m1-integration-bootstrap.txt
results/m1-lint.txt
results/m1-lock-check.txt
results/m1-pretrained-bootstrap.txt
results/m1-python-tests.txt
results/m1-runtime-dependencies.txt
results/m1-summary.json
results/m1-test-discovery-regression.txt
scripts/dev-shell.ps1
scripts/record-environment.ps1
scripts/test.ps1
scripts/verify_m1_artifacts.py
src/model/loader.cpp
src/platform/inspect_main_posix.cpp
src/platform/inspect_main_windows.cpp
src/platform/mapped_file_posix.cpp
src/platform/mapped_file_windows.cpp
src/tools/inspect.cpp
src/tools/inspect.hpp
tests/__init__.py
tests/loader_integration.py
tests/roundtrip.py
tests/test_export.py
tests/test_loader.cpp
tests/test_main.cpp
tests/test_mapped_file.cpp
tests/test_model_format.py
tests/test_support.hpp
uv.lock
```

## Full final test output

<details>
<summary>Python tests</summary>

```text
============================= test session starts =============================
platform win32 -- Python 3.13.15, pytest-8.4.2, pluggy-1.7.0 -- C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime
configfile: pyproject.toml
testpaths: tests
collecting ... collected 20 items

tests/test_export.py::test_conv1d_orientation_and_tied_embedding PASSED  [  5%]
tests/test_export.py::test_untied_output_is_rejected PASSED              [ 10%]
tests/test_export.py::test_unsupported_activation_is_rejected PASSED     [ 15%]
tests/test_export.py::test_golden_matches_direct_pytorch PASSED          [ 20%]
tests/test_model_format.py::test_binary_layout_and_checksums PASSED      [ 25%]
tests/test_model_format.py::test_reject_invalid_tensors[tensor0] PASSED  [ 30%]
tests/test_model_format.py::test_reject_invalid_tensors[tensor1] PASSED  [ 35%]
tests/test_model_format.py::test_reject_invalid_tensors[tensor2] PASSED  [ 40%]
tests/test_model_format.py::test_reject_invalid_tensors[tensor3] PASSED  [ 45%]
tests/test_model_format.py::test_reject_invalid_tensors[tensor4] PASSED  [ 50%]
tests/test_model_format.py::test_reject_invalid_names[] PASSED           [ 55%]
tests/test_model_format.py::test_reject_invalid_names[xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx] PASSED [ 60%]
tests/test_model_format.py::test_reject_invalid_names[embedded\x00null] PASSED [ 65%]
tests/test_model_format.py::test_reject_invalid_names[\u03bb] PASSED     [ 70%]
tests/test_model_format.py::test_reject_invalid_config[config0] PASSED   [ 75%]
tests/test_model_format.py::test_reject_invalid_config[config1] PASSED   [ 80%]
tests/test_model_format.py::test_reject_invalid_config[config2] PASSED   [ 85%]
tests/test_model_format.py::test_reject_invalid_config[config3] PASSED   [ 90%]
tests/test_model_format.py::test_reject_invalid_config[config4] PASSED   [ 95%]
tests/test_model_format.py::test_reject_empty_model PASSED               [100%]

============================= 20 passed in 43.88s =============================

```

</details>

<details>
<summary>Debug CTest output</summary>

```text
UpdateCTestConfiguration  from :C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/DartConfiguration.tcl
UpdateCTestConfiguration  from :C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/DartConfiguration.tcl
Parse Config file:C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/DartConfiguration.tcl
Test project C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 1
    Start 1: unit

1: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\build\debug\nanoinfer_tests.exe"
1: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug
1: Test timeout computed to be: 1500
1: Random seed: 42
1: [==========] Running 12 tests from 3 test suites.
1: [----------] Global test environment set-up.
1: [----------] 5 tests from MappedFile
1: [ RUN      ] MappedFile.ReadsExactBytesAndSupportsUnicodePaths
1: [       OK ] MappedFile.ReadsExactBytesAndSupportsUnicodePaths (7 ms)
1: [ RUN      ] MappedFile.MissingFileFailsCleanly
1: [       OK ] MappedFile.MissingFileFailsCleanly (4 ms)
1: [ RUN      ] MappedFile.EmptyFileFailsCleanly
1: [       OK ] MappedFile.EmptyFileFailsCleanly (5 ms)
1: [ RUN      ] MappedFile.DirectoryFailsCleanly
1: [       OK ] MappedFile.DirectoryFailsCleanly (20 ms)
1: [ RUN      ] MappedFile.MoveTransfersMappingAndAssignmentReleasesPreviousFile
1: [       OK ] MappedFile.MoveTransfersMappingAndAssignmentReleasesPreviousFile (7 ms)
1: [----------] 5 tests from MappedFile (45 ms total)
1:
1: [----------] 1 test from Checksum
1: [ RUN      ] Checksum.MatchesStandardVectorAndChaining
1: [       OK ] Checksum.MatchesStandardVectorAndChaining (0 ms)
1: [----------] 1 test from Checksum (0 ms total)
1:
1: [----------] 6 tests from LoaderTest
1: [ RUN      ] LoaderTest.ReadsShapesStridesAndZeroCopyValues
1: [       OK ] LoaderTest.ReadsShapesStridesAndZeroCopyValues (6 ms)
1: [ RUN      ] LoaderTest.RejectsEveryTruncationBoundary
1: [       OK ] LoaderTest.RejectsEveryTruncationBoundary (940 ms)
1: [ RUN      ] LoaderTest.RejectsBitCorruptionAtEveryByte
1: [       OK ] LoaderTest.RejectsBitCorruptionAtEveryByte (1119 ms)
1: [ RUN      ] LoaderTest.RejectsInvalidMetadataEvenWithCorrectChecksum
1: [       OK ] LoaderTest.RejectsInvalidMetadataEvenWithCorrectChecksum (36 ms)
1: [ RUN      ] LoaderTest.RejectsOverflowOverlapsOffsetsAndTrailingData
1: [       OK ] LoaderTest.RejectsOverflowOverlapsOffsetsAndTrailingData (22 ms)
1: [ RUN      ] LoaderTest.RejectsDuplicateUnterminatedAndInvalidNames
1: [       OK ] LoaderTest.RejectsDuplicateUnterminatedAndInvalidNames (23 ms)
1: [----------] 6 tests from LoaderTest (2148 ms total)
1:
1: [----------] Global test environment tear-down
1: [==========] 12 tests from 3 test suites ran. (2194 ms total)
1: [  PASSED  ] 12 tests.
1/4 Test #1: unit .............................   Passed    5.26 sec
test 2
    Start 2: export_tiny

2: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\.venv\Scripts\python.exe" "-m" "python.export" "--tiny" "--output" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/fixtures/tiny.bin" "--golden" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/fixtures/golden"
2: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime
2: Test timeout computed to be: 1500
2: {
2:   "format_version": 1,
2:   "config": {
2:     "layers": 2,
2:     "d_model": 16,
2:     "heads": 2,
2:     "vocab": 97,
2:     "ctx": 64
2:   },
2:   "tensors": [
2:     {
2:       "name": "wte.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         97,
2:         16
2:       ],
2:       "offset": 3648,
2:       "byte_size": 6208,
2:       "crc32": 1708750088
2:     },
2:     {
2:       "name": "wpe.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         64,
2:         16
2:       ],
2:       "offset": 9856,
2:       "byte_size": 4096,
2:       "crc32": 1388767903
2:     },
2:     {
2:       "name": "h.0.ln_1.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 13952,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.0.ln_1.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 14016,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.0.attn.c_attn.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         48
2:       ],
2:       "offset": 14080,
2:       "byte_size": 3072,
2:       "crc32": 184779411
2:     },
2:     {
2:       "name": "h.0.attn.c_attn.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         48
2:       ],
2:       "offset": 17152,
2:       "byte_size": 192,
2:       "crc32": 2348746994
2:     },
2:     {
2:       "name": "h.0.attn.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         16
2:       ],
2:       "offset": 17344,
2:       "byte_size": 1024,
2:       "crc32": 1610141689
2:     },
2:     {
2:       "name": "h.0.attn.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 18368,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.0.ln_2.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 18432,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.0.ln_2.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 18496,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.0.mlp.c_fc.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         64
2:       ],
2:       "offset": 18560,
2:       "byte_size": 4096,
2:       "crc32": 768724231
2:     },
2:     {
2:       "name": "h.0.mlp.c_fc.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         64
2:       ],
2:       "offset": 22656,
2:       "byte_size": 256,
2:       "crc32": 227968344
2:     },
2:     {
2:       "name": "h.0.mlp.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         64,
2:         16
2:       ],
2:       "offset": 22912,
2:       "byte_size": 4096,
2:       "crc32": 2981679276
2:     },
2:     {
2:       "name": "h.0.mlp.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 27008,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.ln_1.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 27072,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.1.ln_1.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 27136,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.attn.c_attn.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         48
2:       ],
2:       "offset": 27200,
2:       "byte_size": 3072,
2:       "crc32": 3948583692
2:     },
2:     {
2:       "name": "h.1.attn.c_attn.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         48
2:       ],
2:       "offset": 30272,
2:       "byte_size": 192,
2:       "crc32": 2348746994
2:     },
2:     {
2:       "name": "h.1.attn.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         16
2:       ],
2:       "offset": 30464,
2:       "byte_size": 1024,
2:       "crc32": 776059860
2:     },
2:     {
2:       "name": "h.1.attn.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 31488,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.ln_2.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 31552,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.1.ln_2.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 31616,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.mlp.c_fc.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         64
2:       ],
2:       "offset": 31680,
2:       "byte_size": 4096,
2:       "crc32": 4045243238
2:     },
2:     {
2:       "name": "h.1.mlp.c_fc.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         64
2:       ],
2:       "offset": 35776,
2:       "byte_size": 256,
2:       "crc32": 227968344
2:     },
2:     {
2:       "name": "h.1.mlp.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         64,
2:         16
2:       ],
2:       "offset": 36032,
2:       "byte_size": 4096,
2:       "crc32": 320104244
2:     },
2:     {
2:       "name": "h.1.mlp.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 40128,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "ln_f.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 40192,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "ln_f.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 40256,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     }
2:   ],
2:   "source": {
2:     "torch": "2.11.0+cu128",
2:     "transformers": "4.57.1",
2:     "revision": null,
2:     "weight_layout": "Conv1D [in, out]; embeddings [vocab/ctx, d_model]"
2:   }
2: }
2/4 Test #2: export_tiny ......................   Passed   22.88 sec
test 3
    Start 3: roundtrip_tiny

3: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\.venv\Scripts\python.exe" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/tests/roundtrip.py" "--inspector" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/gpt2inspect.exe" "--model" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/fixtures/tiny.bin" "--manifest" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/fixtures/tiny.json"
3: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug
3: Test timeout computed to be: 1500
3: PASS wte.weight shape=[97, 16] crc32=65d97508
3: PASS wpe.weight shape=[64, 16] crc32=52c6ea9f
3: PASS h.0.ln_1.weight shape=[16] crc32=f700a42a
3: PASS h.0.ln_1.bias shape=[16] crc32=758d6336
3: PASS h.0.attn.c_attn.weight shape=[16, 48] crc32=0b038293
3: PASS h.0.attn.c_attn.bias shape=[48] crc32=8bff08f2
3: PASS h.0.attn.c_proj.weight shape=[16, 16] crc32=5ff8cff9
3: PASS h.0.attn.c_proj.bias shape=[16] crc32=758d6336
3: PASS h.0.ln_2.weight shape=[16] crc32=f700a42a
3: PASS h.0.ln_2.bias shape=[16] crc32=758d6336
3: PASS h.0.mlp.c_fc.weight shape=[16, 64] crc32=2dd1cd07
3: PASS h.0.mlp.c_fc.bias shape=[64] crc32=0d968558
3: PASS h.0.mlp.c_proj.weight shape=[64, 16] crc32=b1b8d0ac
3: PASS h.0.mlp.c_proj.bias shape=[16] crc32=758d6336
3: PASS h.1.ln_1.weight shape=[16] crc32=f700a42a
3: PASS h.1.ln_1.bias shape=[16] crc32=758d6336
3: PASS h.1.attn.c_attn.weight shape=[16, 48] crc32=eb5a9b0c
3: PASS h.1.attn.c_attn.bias shape=[48] crc32=8bff08f2
3: PASS h.1.attn.c_proj.weight shape=[16, 16] crc32=2e41bbd4
3: PASS h.1.attn.c_proj.bias shape=[16] crc32=758d6336
3: PASS h.1.ln_2.weight shape=[16] crc32=f700a42a
3: PASS h.1.ln_2.bias shape=[16] crc32=758d6336
3: PASS h.1.mlp.c_fc.weight shape=[16, 64] crc32=f11d8366
3: PASS h.1.mlp.c_fc.bias shape=[64] crc32=0d968558
3: PASS h.1.mlp.c_proj.weight shape=[64, 16] crc32=13146734
3: PASS h.1.mlp.c_proj.bias shape=[16] crc32=758d6336
3: PASS ln_f.weight shape=[16] crc32=f700a42a
3: PASS ln_f.bias shape=[16] crc32=758d6336
3: PASS all 28 tensor shapes, checksums, offsets and sizes
3/4 Test #3: roundtrip_tiny ...................   Passed    3.42 sec
test 4
    Start 4: loader_integration

4: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\.venv\Scripts\python.exe" "-m" "tests.loader_integration" "--inspector" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/gpt2inspect.exe" "--directory" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/debug/awkward"
4: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime
4: Test timeout computed to be: 1500
4: Random seed: 42
4: PASS 32 random/awkward tensors, all ranks, Unicode path and JSON escaping
4: PASS clean error for empty: model error: model file is empty or too large
4: PASS clean error for header: model error: truncated header
4: PASS clean error for table: model error: invalid tensor count
4: PASS clean error for payload: model error: file length mismatch
4: PASS clean error for trailing: model error: file length mismatch
4: PASS clean error for corrupt: model error: tensor checksum mismatch
4: PASS missing CLI argument
4/4 Test #4: loader_integration ...............   Passed    2.84 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =  34.43 sec

```

</details>

<details>
<summary>Debug pretrained round-trip</summary>

```text
PASS wte.weight shape=[50257, 768] crc32=ce20fa37
PASS wpe.weight shape=[1024, 768] crc32=87774e0a
PASS h.0.ln_1.weight shape=[768] crc32=a3ab5385
PASS h.0.ln_1.bias shape=[768] crc32=5acb5896
PASS h.0.attn.c_attn.weight shape=[768, 2304] crc32=c424d078
PASS h.0.attn.c_attn.bias shape=[2304] crc32=86271fa1
PASS h.0.attn.c_proj.weight shape=[768, 768] crc32=ad37c738
PASS h.0.attn.c_proj.bias shape=[768] crc32=264b8415
PASS h.0.ln_2.weight shape=[768] crc32=6c6a9da9
PASS h.0.ln_2.bias shape=[768] crc32=eef867d0
PASS h.0.mlp.c_fc.weight shape=[768, 3072] crc32=f9135263
PASS h.0.mlp.c_fc.bias shape=[3072] crc32=b31704ba
PASS h.0.mlp.c_proj.weight shape=[3072, 768] crc32=4d4471e5
PASS h.0.mlp.c_proj.bias shape=[768] crc32=a1aebdff
PASS h.1.ln_1.weight shape=[768] crc32=90c32cba
PASS h.1.ln_1.bias shape=[768] crc32=335167ae
PASS h.1.attn.c_attn.weight shape=[768, 2304] crc32=0482147d
PASS h.1.attn.c_attn.bias shape=[2304] crc32=84797379
PASS h.1.attn.c_proj.weight shape=[768, 768] crc32=fd97f472
PASS h.1.attn.c_proj.bias shape=[768] crc32=ab0d7d64
PASS h.1.ln_2.weight shape=[768] crc32=0391b457
PASS h.1.ln_2.bias shape=[768] crc32=1f595699
PASS h.1.mlp.c_fc.weight shape=[768, 3072] crc32=92b9be7b
PASS h.1.mlp.c_fc.bias shape=[3072] crc32=af3c6a31
PASS h.1.mlp.c_proj.weight shape=[3072, 768] crc32=166f6faa
PASS h.1.mlp.c_proj.bias shape=[768] crc32=0727d3a7
PASS h.2.ln_1.weight shape=[768] crc32=614e5b4c
PASS h.2.ln_1.bias shape=[768] crc32=6f8666f8
PASS h.2.attn.c_attn.weight shape=[768, 2304] crc32=46171872
PASS h.2.attn.c_attn.bias shape=[2304] crc32=773d49bd
PASS h.2.attn.c_proj.weight shape=[768, 768] crc32=20d25c19
PASS h.2.attn.c_proj.bias shape=[768] crc32=9cb919e2
PASS h.2.ln_2.weight shape=[768] crc32=5e4ad63e
PASS h.2.ln_2.bias shape=[768] crc32=af72a18a
PASS h.2.mlp.c_fc.weight shape=[768, 3072] crc32=6ac9ce3f
PASS h.2.mlp.c_fc.bias shape=[3072] crc32=3f518ab9
PASS h.2.mlp.c_proj.weight shape=[3072, 768] crc32=190ea8e1
PASS h.2.mlp.c_proj.bias shape=[768] crc32=1aff4d2d
PASS h.3.ln_1.weight shape=[768] crc32=b7ede68e
PASS h.3.ln_1.bias shape=[768] crc32=3ad9178a
PASS h.3.attn.c_attn.weight shape=[768, 2304] crc32=83cd7597
PASS h.3.attn.c_attn.bias shape=[2304] crc32=c4cd1d4d
PASS h.3.attn.c_proj.weight shape=[768, 768] crc32=a54f3ac6
PASS h.3.attn.c_proj.bias shape=[768] crc32=b8a25d21
PASS h.3.ln_2.weight shape=[768] crc32=ff85c704
PASS h.3.ln_2.bias shape=[768] crc32=d609641e
PASS h.3.mlp.c_fc.weight shape=[768, 3072] crc32=c43f7830
PASS h.3.mlp.c_fc.bias shape=[3072] crc32=c5ac67fe
PASS h.3.mlp.c_proj.weight shape=[3072, 768] crc32=a69e2d26
PASS h.3.mlp.c_proj.bias shape=[768] crc32=6ecaf5ba
PASS h.4.ln_1.weight shape=[768] crc32=3a0f6a1f
PASS h.4.ln_1.bias shape=[768] crc32=49a72efe
PASS h.4.attn.c_attn.weight shape=[768, 2304] crc32=574e265b
PASS h.4.attn.c_attn.bias shape=[2304] crc32=391b6026
PASS h.4.attn.c_proj.weight shape=[768, 768] crc32=eaea11c1
PASS h.4.attn.c_proj.bias shape=[768] crc32=6fd56fa9
PASS h.4.ln_2.weight shape=[768] crc32=39e05e1b
PASS h.4.ln_2.bias shape=[768] crc32=bffa0c1a
PASS h.4.mlp.c_fc.weight shape=[768, 3072] crc32=eec40958
PASS h.4.mlp.c_fc.bias shape=[3072] crc32=032f5160
PASS h.4.mlp.c_proj.weight shape=[3072, 768] crc32=cfeb2fb7
PASS h.4.mlp.c_proj.bias shape=[768] crc32=f72470b4
PASS h.5.ln_1.weight shape=[768] crc32=2cd9f6c5
PASS h.5.ln_1.bias shape=[768] crc32=ee7f9d2e
PASS h.5.attn.c_attn.weight shape=[768, 2304] crc32=b093cc4b
PASS h.5.attn.c_attn.bias shape=[2304] crc32=94d5a1ad
PASS h.5.attn.c_proj.weight shape=[768, 768] crc32=6a241570
PASS h.5.attn.c_proj.bias shape=[768] crc32=69dad12f
PASS h.5.ln_2.weight shape=[768] crc32=b3c361d2
PASS h.5.ln_2.bias shape=[768] crc32=465a46ac
PASS h.5.mlp.c_fc.weight shape=[768, 3072] crc32=c9c3215b
PASS h.5.mlp.c_fc.bias shape=[3072] crc32=9f234310
PASS h.5.mlp.c_proj.weight shape=[3072, 768] crc32=4a681c55
PASS h.5.mlp.c_proj.bias shape=[768] crc32=9f5510b3
PASS h.6.ln_1.weight shape=[768] crc32=3292ee8f
PASS h.6.ln_1.bias shape=[768] crc32=59e21e2c
PASS h.6.attn.c_attn.weight shape=[768, 2304] crc32=802505e1
PASS h.6.attn.c_attn.bias shape=[2304] crc32=30453ea8
PASS h.6.attn.c_proj.weight shape=[768, 768] crc32=f04fac3f
PASS h.6.attn.c_proj.bias shape=[768] crc32=88340e3c
PASS h.6.ln_2.weight shape=[768] crc32=7a30953d
PASS h.6.ln_2.bias shape=[768] crc32=d3820861
PASS h.6.mlp.c_fc.weight shape=[768, 3072] crc32=45638738
PASS h.6.mlp.c_fc.bias shape=[3072] crc32=17f8102e
PASS h.6.mlp.c_proj.weight shape=[3072, 768] crc32=95b16dd3
PASS h.6.mlp.c_proj.bias shape=[768] crc32=33cb7afd
PASS h.7.ln_1.weight shape=[768] crc32=53998f37
PASS h.7.ln_1.bias shape=[768] crc32=cfddbdd2
PASS h.7.attn.c_attn.weight shape=[768, 2304] crc32=5598d1ed
PASS h.7.attn.c_attn.bias shape=[2304] crc32=d15df1eb
PASS h.7.attn.c_proj.weight shape=[768, 768] crc32=b4c9ae4b
PASS h.7.attn.c_proj.bias shape=[768] crc32=19adf4e3
PASS h.7.ln_2.weight shape=[768] crc32=72b7b981
PASS h.7.ln_2.bias shape=[768] crc32=f6dfd17e
PASS h.7.mlp.c_fc.weight shape=[768, 3072] crc32=52fb3911
PASS h.7.mlp.c_fc.bias shape=[3072] crc32=f373fd70
PASS h.7.mlp.c_proj.weight shape=[3072, 768] crc32=0119b185
PASS h.7.mlp.c_proj.bias shape=[768] crc32=4e944f13
PASS h.8.ln_1.weight shape=[768] crc32=671c8a63
PASS h.8.ln_1.bias shape=[768] crc32=826624ed
PASS h.8.attn.c_attn.weight shape=[768, 2304] crc32=1f3ce922
PASS h.8.attn.c_attn.bias shape=[2304] crc32=feb6b19f
PASS h.8.attn.c_proj.weight shape=[768, 768] crc32=2637d846
PASS h.8.attn.c_proj.bias shape=[768] crc32=072ff038
PASS h.8.ln_2.weight shape=[768] crc32=cf654be1
PASS h.8.ln_2.bias shape=[768] crc32=ff488087
PASS h.8.mlp.c_fc.weight shape=[768, 3072] crc32=3981d65f
PASS h.8.mlp.c_fc.bias shape=[3072] crc32=339bd418
PASS h.8.mlp.c_proj.weight shape=[3072, 768] crc32=27a3e65a
PASS h.8.mlp.c_proj.bias shape=[768] crc32=61227b3b
PASS h.9.ln_1.weight shape=[768] crc32=2f03e0f7
PASS h.9.ln_1.bias shape=[768] crc32=10d6006a
PASS h.9.attn.c_attn.weight shape=[768, 2304] crc32=40b16e51
PASS h.9.attn.c_attn.bias shape=[2304] crc32=efd2ece5
PASS h.9.attn.c_proj.weight shape=[768, 768] crc32=66c74f7f
PASS h.9.attn.c_proj.bias shape=[768] crc32=6d48b2cf
PASS h.9.ln_2.weight shape=[768] crc32=1f25236b
PASS h.9.ln_2.bias shape=[768] crc32=e475ece5
PASS h.9.mlp.c_fc.weight shape=[768, 3072] crc32=ddaf6ecf
PASS h.9.mlp.c_fc.bias shape=[3072] crc32=d680291d
PASS h.9.mlp.c_proj.weight shape=[3072, 768] crc32=312e436e
PASS h.9.mlp.c_proj.bias shape=[768] crc32=5c189b80
PASS h.10.ln_1.weight shape=[768] crc32=755a3ba6
PASS h.10.ln_1.bias shape=[768] crc32=c66d8f55
PASS h.10.attn.c_attn.weight shape=[768, 2304] crc32=5e08bf63
PASS h.10.attn.c_attn.bias shape=[2304] crc32=7575b548
PASS h.10.attn.c_proj.weight shape=[768, 768] crc32=3c774cfe
PASS h.10.attn.c_proj.bias shape=[768] crc32=f254f310
PASS h.10.ln_2.weight shape=[768] crc32=8343c266
PASS h.10.ln_2.bias shape=[768] crc32=eac1e8d6
PASS h.10.mlp.c_fc.weight shape=[768, 3072] crc32=702857b0
PASS h.10.mlp.c_fc.bias shape=[3072] crc32=761e513e
PASS h.10.mlp.c_proj.weight shape=[3072, 768] crc32=d34573a8
PASS h.10.mlp.c_proj.bias shape=[768] crc32=4af7f176
PASS h.11.ln_1.weight shape=[768] crc32=d265810e
PASS h.11.ln_1.bias shape=[768] crc32=64a76217
PASS h.11.attn.c_attn.weight shape=[768, 2304] crc32=011a3c30
PASS h.11.attn.c_attn.bias shape=[2304] crc32=95e9105c
PASS h.11.attn.c_proj.weight shape=[768, 768] crc32=0693462e
PASS h.11.attn.c_proj.bias shape=[768] crc32=1ca96691
PASS h.11.ln_2.weight shape=[768] crc32=5894078f
PASS h.11.ln_2.bias shape=[768] crc32=3e3eca73
PASS h.11.mlp.c_fc.weight shape=[768, 3072] crc32=ad288899
PASS h.11.mlp.c_fc.bias shape=[3072] crc32=a858a7e8
PASS h.11.mlp.c_proj.weight shape=[3072, 768] crc32=b41639c2
PASS h.11.mlp.c_proj.bias shape=[768] crc32=d19c1af9
PASS ln_f.weight shape=[768] crc32=66271b90
PASS ln_f.bias shape=[768] crc32=71261dab
PASS all 148 tensor shapes, checksums, offsets and sizes

```

</details>

<details>
<summary>AddressSanitizer CTest output</summary>

```text
UpdateCTestConfiguration  from :C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/DartConfiguration.tcl
UpdateCTestConfiguration  from :C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/DartConfiguration.tcl
Parse Config file:C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/DartConfiguration.tcl
Test project C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan
Constructing a list of tests
Done constructing a list of tests
Updating test list for fixtures
Added 0 tests to meet fixture requirements
Checking test dependency graph...
Checking test dependency graph end
test 1
    Start 1: unit

1: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\build\asan\nanoinfer_tests.exe"
1: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan
1: Test timeout computed to be: 1500
1: Random seed: 42
1: [==========] Running 12 tests from 3 test suites.
1: [----------] Global test environment set-up.
1: [----------] 5 tests from MappedFile
1: [ RUN      ] MappedFile.ReadsExactBytesAndSupportsUnicodePaths
1: [       OK ] MappedFile.ReadsExactBytesAndSupportsUnicodePaths (9 ms)
1: [ RUN      ] MappedFile.MissingFileFailsCleanly
1: [       OK ] MappedFile.MissingFileFailsCleanly (5 ms)
1: [ RUN      ] MappedFile.EmptyFileFailsCleanly
1: [       OK ] MappedFile.EmptyFileFailsCleanly (7 ms)
1: [ RUN      ] MappedFile.DirectoryFailsCleanly
1: [       OK ] MappedFile.DirectoryFailsCleanly (6 ms)
1: [ RUN      ] MappedFile.MoveTransfersMappingAndAssignmentReleasesPreviousFile
1: [       OK ] MappedFile.MoveTransfersMappingAndAssignmentReleasesPreviousFile (8 ms)
1: [----------] 5 tests from MappedFile (38 ms total)
1:
1: [----------] 1 test from Checksum
1: [ RUN      ] Checksum.MatchesStandardVectorAndChaining
1: [       OK ] Checksum.MatchesStandardVectorAndChaining (0 ms)
1: [----------] 1 test from Checksum (0 ms total)
1:
1: [----------] 6 tests from LoaderTest
1: [ RUN      ] LoaderTest.ReadsShapesStridesAndZeroCopyValues
1: [       OK ] LoaderTest.ReadsShapesStridesAndZeroCopyValues (9 ms)
1: [ RUN      ] LoaderTest.RejectsEveryTruncationBoundary
1: [       OK ] LoaderTest.RejectsEveryTruncationBoundary (1071 ms)
1: [ RUN      ] LoaderTest.RejectsBitCorruptionAtEveryByte
1: [       OK ] LoaderTest.RejectsBitCorruptionAtEveryByte (1141 ms)
1: [ RUN      ] LoaderTest.RejectsInvalidMetadataEvenWithCorrectChecksum
1: [       OK ] LoaderTest.RejectsInvalidMetadataEvenWithCorrectChecksum (38 ms)
1: [ RUN      ] LoaderTest.RejectsOverflowOverlapsOffsetsAndTrailingData
1: [       OK ] LoaderTest.RejectsOverflowOverlapsOffsetsAndTrailingData (28 ms)
1: [ RUN      ] LoaderTest.RejectsDuplicateUnterminatedAndInvalidNames
1: [       OK ] LoaderTest.RejectsDuplicateUnterminatedAndInvalidNames (16 ms)
1: [----------] 6 tests from LoaderTest (2305 ms total)
1:
1: [----------] Global test environment tear-down
1: [==========] 12 tests from 3 test suites ran. (2346 ms total)
1: [  PASSED  ] 12 tests.
1/4 Test #1: unit .............................   Passed    6.37 sec
test 2
    Start 2: export_tiny

2: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\.venv\Scripts\python.exe" "-m" "python.export" "--tiny" "--output" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/fixtures/tiny.bin" "--golden" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/fixtures/golden"
2: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime
2: Test timeout computed to be: 1500
2: {
2:   "format_version": 1,
2:   "config": {
2:     "layers": 2,
2:     "d_model": 16,
2:     "heads": 2,
2:     "vocab": 97,
2:     "ctx": 64
2:   },
2:   "tensors": [
2:     {
2:       "name": "wte.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         97,
2:         16
2:       ],
2:       "offset": 3648,
2:       "byte_size": 6208,
2:       "crc32": 1708750088
2:     },
2:     {
2:       "name": "wpe.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         64,
2:         16
2:       ],
2:       "offset": 9856,
2:       "byte_size": 4096,
2:       "crc32": 1388767903
2:     },
2:     {
2:       "name": "h.0.ln_1.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 13952,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.0.ln_1.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 14016,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.0.attn.c_attn.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         48
2:       ],
2:       "offset": 14080,
2:       "byte_size": 3072,
2:       "crc32": 184779411
2:     },
2:     {
2:       "name": "h.0.attn.c_attn.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         48
2:       ],
2:       "offset": 17152,
2:       "byte_size": 192,
2:       "crc32": 2348746994
2:     },
2:     {
2:       "name": "h.0.attn.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         16
2:       ],
2:       "offset": 17344,
2:       "byte_size": 1024,
2:       "crc32": 1610141689
2:     },
2:     {
2:       "name": "h.0.attn.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 18368,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.0.ln_2.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 18432,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.0.ln_2.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 18496,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.0.mlp.c_fc.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         64
2:       ],
2:       "offset": 18560,
2:       "byte_size": 4096,
2:       "crc32": 768724231
2:     },
2:     {
2:       "name": "h.0.mlp.c_fc.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         64
2:       ],
2:       "offset": 22656,
2:       "byte_size": 256,
2:       "crc32": 227968344
2:     },
2:     {
2:       "name": "h.0.mlp.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         64,
2:         16
2:       ],
2:       "offset": 22912,
2:       "byte_size": 4096,
2:       "crc32": 2981679276
2:     },
2:     {
2:       "name": "h.0.mlp.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 27008,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.ln_1.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 27072,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.1.ln_1.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 27136,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.attn.c_attn.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         48
2:       ],
2:       "offset": 27200,
2:       "byte_size": 3072,
2:       "crc32": 3948583692
2:     },
2:     {
2:       "name": "h.1.attn.c_attn.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         48
2:       ],
2:       "offset": 30272,
2:       "byte_size": 192,
2:       "crc32": 2348746994
2:     },
2:     {
2:       "name": "h.1.attn.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         16
2:       ],
2:       "offset": 30464,
2:       "byte_size": 1024,
2:       "crc32": 776059860
2:     },
2:     {
2:       "name": "h.1.attn.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 31488,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.ln_2.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 31552,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "h.1.ln_2.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 31616,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "h.1.mlp.c_fc.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16,
2:         64
2:       ],
2:       "offset": 31680,
2:       "byte_size": 4096,
2:       "crc32": 4045243238
2:     },
2:     {
2:       "name": "h.1.mlp.c_fc.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         64
2:       ],
2:       "offset": 35776,
2:       "byte_size": 256,
2:       "crc32": 227968344
2:     },
2:     {
2:       "name": "h.1.mlp.c_proj.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         64,
2:         16
2:       ],
2:       "offset": 36032,
2:       "byte_size": 4096,
2:       "crc32": 320104244
2:     },
2:     {
2:       "name": "h.1.mlp.c_proj.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 40128,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     },
2:     {
2:       "name": "ln_f.weight",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 40192,
2:       "byte_size": 64,
2:       "crc32": 4144014378
2:     },
2:     {
2:       "name": "ln_f.bias",
2:       "dtype": "f32",
2:       "shape": [
2:         16
2:       ],
2:       "offset": 40256,
2:       "byte_size": 64,
2:       "crc32": 1972200246
2:     }
2:   ],
2:   "source": {
2:     "torch": "2.11.0+cu128",
2:     "transformers": "4.57.1",
2:     "revision": null,
2:     "weight_layout": "Conv1D [in, out]; embeddings [vocab/ctx, d_model]"
2:   }
2: }
2/4 Test #2: export_tiny ......................   Passed   20.64 sec
test 3
    Start 3: roundtrip_tiny

3: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\.venv\Scripts\python.exe" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/tests/roundtrip.py" "--inspector" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/gpt2inspect.exe" "--model" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/fixtures/tiny.bin" "--manifest" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/fixtures/tiny.json"
3: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan
3: Test timeout computed to be: 1500
3: PASS wte.weight shape=[97, 16] crc32=65d97508
3: PASS wpe.weight shape=[64, 16] crc32=52c6ea9f
3: PASS h.0.ln_1.weight shape=[16] crc32=f700a42a
3: PASS h.0.ln_1.bias shape=[16] crc32=758d6336
3: PASS h.0.attn.c_attn.weight shape=[16, 48] crc32=0b038293
3: PASS h.0.attn.c_attn.bias shape=[48] crc32=8bff08f2
3: PASS h.0.attn.c_proj.weight shape=[16, 16] crc32=5ff8cff9
3: PASS h.0.attn.c_proj.bias shape=[16] crc32=758d6336
3: PASS h.0.ln_2.weight shape=[16] crc32=f700a42a
3: PASS h.0.ln_2.bias shape=[16] crc32=758d6336
3: PASS h.0.mlp.c_fc.weight shape=[16, 64] crc32=2dd1cd07
3: PASS h.0.mlp.c_fc.bias shape=[64] crc32=0d968558
3: PASS h.0.mlp.c_proj.weight shape=[64, 16] crc32=b1b8d0ac
3: PASS h.0.mlp.c_proj.bias shape=[16] crc32=758d6336
3: PASS h.1.ln_1.weight shape=[16] crc32=f700a42a
3: PASS h.1.ln_1.bias shape=[16] crc32=758d6336
3: PASS h.1.attn.c_attn.weight shape=[16, 48] crc32=eb5a9b0c
3: PASS h.1.attn.c_attn.bias shape=[48] crc32=8bff08f2
3: PASS h.1.attn.c_proj.weight shape=[16, 16] crc32=2e41bbd4
3: PASS h.1.attn.c_proj.bias shape=[16] crc32=758d6336
3: PASS h.1.ln_2.weight shape=[16] crc32=f700a42a
3: PASS h.1.ln_2.bias shape=[16] crc32=758d6336
3: PASS h.1.mlp.c_fc.weight shape=[16, 64] crc32=f11d8366
3: PASS h.1.mlp.c_fc.bias shape=[64] crc32=0d968558
3: PASS h.1.mlp.c_proj.weight shape=[64, 16] crc32=13146734
3: PASS h.1.mlp.c_proj.bias shape=[16] crc32=758d6336
3: PASS ln_f.weight shape=[16] crc32=f700a42a
3: PASS ln_f.bias shape=[16] crc32=758d6336
3: PASS all 28 tensor shapes, checksums, offsets and sizes
3/4 Test #3: roundtrip_tiny ...................   Passed    3.56 sec
test 4
    Start 4: loader_integration

4: Test command: "C:\Users\arnav\Downloads\SDE PROJs\cpu-inference-runtime\.venv\Scripts\python.exe" "-m" "tests.loader_integration" "--inspector" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/gpt2inspect.exe" "--directory" "C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime/build/asan/awkward"
4: Working Directory: C:/Users/arnav/Downloads/SDE PROJs/cpu-inference-runtime
4: Test timeout computed to be: 1500
4: Random seed: 42
4: PASS 32 random/awkward tensors, all ranks, Unicode path and JSON escaping
4: PASS clean error for empty: model error: model file is empty or too large
4: PASS clean error for header: model error: truncated header
4: PASS clean error for table: model error: invalid tensor count
4: PASS clean error for payload: model error: file length mismatch
4: PASS clean error for trailing: model error: file length mismatch
4: PASS clean error for corrupt: model error: tensor checksum mismatch
4: PASS missing CLI argument
4/4 Test #4: loader_integration ...............   Passed    4.77 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =  35.37 sec

```

</details>

<details>
<summary>AddressSanitizer pretrained round-trip</summary>

```text
PASS wte.weight shape=[50257, 768] crc32=ce20fa37
PASS wpe.weight shape=[1024, 768] crc32=87774e0a
PASS h.0.ln_1.weight shape=[768] crc32=a3ab5385
PASS h.0.ln_1.bias shape=[768] crc32=5acb5896
PASS h.0.attn.c_attn.weight shape=[768, 2304] crc32=c424d078
PASS h.0.attn.c_attn.bias shape=[2304] crc32=86271fa1
PASS h.0.attn.c_proj.weight shape=[768, 768] crc32=ad37c738
PASS h.0.attn.c_proj.bias shape=[768] crc32=264b8415
PASS h.0.ln_2.weight shape=[768] crc32=6c6a9da9
PASS h.0.ln_2.bias shape=[768] crc32=eef867d0
PASS h.0.mlp.c_fc.weight shape=[768, 3072] crc32=f9135263
PASS h.0.mlp.c_fc.bias shape=[3072] crc32=b31704ba
PASS h.0.mlp.c_proj.weight shape=[3072, 768] crc32=4d4471e5
PASS h.0.mlp.c_proj.bias shape=[768] crc32=a1aebdff
PASS h.1.ln_1.weight shape=[768] crc32=90c32cba
PASS h.1.ln_1.bias shape=[768] crc32=335167ae
PASS h.1.attn.c_attn.weight shape=[768, 2304] crc32=0482147d
PASS h.1.attn.c_attn.bias shape=[2304] crc32=84797379
PASS h.1.attn.c_proj.weight shape=[768, 768] crc32=fd97f472
PASS h.1.attn.c_proj.bias shape=[768] crc32=ab0d7d64
PASS h.1.ln_2.weight shape=[768] crc32=0391b457
PASS h.1.ln_2.bias shape=[768] crc32=1f595699
PASS h.1.mlp.c_fc.weight shape=[768, 3072] crc32=92b9be7b
PASS h.1.mlp.c_fc.bias shape=[3072] crc32=af3c6a31
PASS h.1.mlp.c_proj.weight shape=[3072, 768] crc32=166f6faa
PASS h.1.mlp.c_proj.bias shape=[768] crc32=0727d3a7
PASS h.2.ln_1.weight shape=[768] crc32=614e5b4c
PASS h.2.ln_1.bias shape=[768] crc32=6f8666f8
PASS h.2.attn.c_attn.weight shape=[768, 2304] crc32=46171872
PASS h.2.attn.c_attn.bias shape=[2304] crc32=773d49bd
PASS h.2.attn.c_proj.weight shape=[768, 768] crc32=20d25c19
PASS h.2.attn.c_proj.bias shape=[768] crc32=9cb919e2
PASS h.2.ln_2.weight shape=[768] crc32=5e4ad63e
PASS h.2.ln_2.bias shape=[768] crc32=af72a18a
PASS h.2.mlp.c_fc.weight shape=[768, 3072] crc32=6ac9ce3f
PASS h.2.mlp.c_fc.bias shape=[3072] crc32=3f518ab9
PASS h.2.mlp.c_proj.weight shape=[3072, 768] crc32=190ea8e1
PASS h.2.mlp.c_proj.bias shape=[768] crc32=1aff4d2d
PASS h.3.ln_1.weight shape=[768] crc32=b7ede68e
PASS h.3.ln_1.bias shape=[768] crc32=3ad9178a
PASS h.3.attn.c_attn.weight shape=[768, 2304] crc32=83cd7597
PASS h.3.attn.c_attn.bias shape=[2304] crc32=c4cd1d4d
PASS h.3.attn.c_proj.weight shape=[768, 768] crc32=a54f3ac6
PASS h.3.attn.c_proj.bias shape=[768] crc32=b8a25d21
PASS h.3.ln_2.weight shape=[768] crc32=ff85c704
PASS h.3.ln_2.bias shape=[768] crc32=d609641e
PASS h.3.mlp.c_fc.weight shape=[768, 3072] crc32=c43f7830
PASS h.3.mlp.c_fc.bias shape=[3072] crc32=c5ac67fe
PASS h.3.mlp.c_proj.weight shape=[3072, 768] crc32=a69e2d26
PASS h.3.mlp.c_proj.bias shape=[768] crc32=6ecaf5ba
PASS h.4.ln_1.weight shape=[768] crc32=3a0f6a1f
PASS h.4.ln_1.bias shape=[768] crc32=49a72efe
PASS h.4.attn.c_attn.weight shape=[768, 2304] crc32=574e265b
PASS h.4.attn.c_attn.bias shape=[2304] crc32=391b6026
PASS h.4.attn.c_proj.weight shape=[768, 768] crc32=eaea11c1
PASS h.4.attn.c_proj.bias shape=[768] crc32=6fd56fa9
PASS h.4.ln_2.weight shape=[768] crc32=39e05e1b
PASS h.4.ln_2.bias shape=[768] crc32=bffa0c1a
PASS h.4.mlp.c_fc.weight shape=[768, 3072] crc32=eec40958
PASS h.4.mlp.c_fc.bias shape=[3072] crc32=032f5160
PASS h.4.mlp.c_proj.weight shape=[3072, 768] crc32=cfeb2fb7
PASS h.4.mlp.c_proj.bias shape=[768] crc32=f72470b4
PASS h.5.ln_1.weight shape=[768] crc32=2cd9f6c5
PASS h.5.ln_1.bias shape=[768] crc32=ee7f9d2e
PASS h.5.attn.c_attn.weight shape=[768, 2304] crc32=b093cc4b
PASS h.5.attn.c_attn.bias shape=[2304] crc32=94d5a1ad
PASS h.5.attn.c_proj.weight shape=[768, 768] crc32=6a241570
PASS h.5.attn.c_proj.bias shape=[768] crc32=69dad12f
PASS h.5.ln_2.weight shape=[768] crc32=b3c361d2
PASS h.5.ln_2.bias shape=[768] crc32=465a46ac
PASS h.5.mlp.c_fc.weight shape=[768, 3072] crc32=c9c3215b
PASS h.5.mlp.c_fc.bias shape=[3072] crc32=9f234310
PASS h.5.mlp.c_proj.weight shape=[3072, 768] crc32=4a681c55
PASS h.5.mlp.c_proj.bias shape=[768] crc32=9f5510b3
PASS h.6.ln_1.weight shape=[768] crc32=3292ee8f
PASS h.6.ln_1.bias shape=[768] crc32=59e21e2c
PASS h.6.attn.c_attn.weight shape=[768, 2304] crc32=802505e1
PASS h.6.attn.c_attn.bias shape=[2304] crc32=30453ea8
PASS h.6.attn.c_proj.weight shape=[768, 768] crc32=f04fac3f
PASS h.6.attn.c_proj.bias shape=[768] crc32=88340e3c
PASS h.6.ln_2.weight shape=[768] crc32=7a30953d
PASS h.6.ln_2.bias shape=[768] crc32=d3820861
PASS h.6.mlp.c_fc.weight shape=[768, 3072] crc32=45638738
PASS h.6.mlp.c_fc.bias shape=[3072] crc32=17f8102e
PASS h.6.mlp.c_proj.weight shape=[3072, 768] crc32=95b16dd3
PASS h.6.mlp.c_proj.bias shape=[768] crc32=33cb7afd
PASS h.7.ln_1.weight shape=[768] crc32=53998f37
PASS h.7.ln_1.bias shape=[768] crc32=cfddbdd2
PASS h.7.attn.c_attn.weight shape=[768, 2304] crc32=5598d1ed
PASS h.7.attn.c_attn.bias shape=[2304] crc32=d15df1eb
PASS h.7.attn.c_proj.weight shape=[768, 768] crc32=b4c9ae4b
PASS h.7.attn.c_proj.bias shape=[768] crc32=19adf4e3
PASS h.7.ln_2.weight shape=[768] crc32=72b7b981
PASS h.7.ln_2.bias shape=[768] crc32=f6dfd17e
PASS h.7.mlp.c_fc.weight shape=[768, 3072] crc32=52fb3911
PASS h.7.mlp.c_fc.bias shape=[3072] crc32=f373fd70
PASS h.7.mlp.c_proj.weight shape=[3072, 768] crc32=0119b185
PASS h.7.mlp.c_proj.bias shape=[768] crc32=4e944f13
PASS h.8.ln_1.weight shape=[768] crc32=671c8a63
PASS h.8.ln_1.bias shape=[768] crc32=826624ed
PASS h.8.attn.c_attn.weight shape=[768, 2304] crc32=1f3ce922
PASS h.8.attn.c_attn.bias shape=[2304] crc32=feb6b19f
PASS h.8.attn.c_proj.weight shape=[768, 768] crc32=2637d846
PASS h.8.attn.c_proj.bias shape=[768] crc32=072ff038
PASS h.8.ln_2.weight shape=[768] crc32=cf654be1
PASS h.8.ln_2.bias shape=[768] crc32=ff488087
PASS h.8.mlp.c_fc.weight shape=[768, 3072] crc32=3981d65f
PASS h.8.mlp.c_fc.bias shape=[3072] crc32=339bd418
PASS h.8.mlp.c_proj.weight shape=[3072, 768] crc32=27a3e65a
PASS h.8.mlp.c_proj.bias shape=[768] crc32=61227b3b
PASS h.9.ln_1.weight shape=[768] crc32=2f03e0f7
PASS h.9.ln_1.bias shape=[768] crc32=10d6006a
PASS h.9.attn.c_attn.weight shape=[768, 2304] crc32=40b16e51
PASS h.9.attn.c_attn.bias shape=[2304] crc32=efd2ece5
PASS h.9.attn.c_proj.weight shape=[768, 768] crc32=66c74f7f
PASS h.9.attn.c_proj.bias shape=[768] crc32=6d48b2cf
PASS h.9.ln_2.weight shape=[768] crc32=1f25236b
PASS h.9.ln_2.bias shape=[768] crc32=e475ece5
PASS h.9.mlp.c_fc.weight shape=[768, 3072] crc32=ddaf6ecf
PASS h.9.mlp.c_fc.bias shape=[3072] crc32=d680291d
PASS h.9.mlp.c_proj.weight shape=[3072, 768] crc32=312e436e
PASS h.9.mlp.c_proj.bias shape=[768] crc32=5c189b80
PASS h.10.ln_1.weight shape=[768] crc32=755a3ba6
PASS h.10.ln_1.bias shape=[768] crc32=c66d8f55
PASS h.10.attn.c_attn.weight shape=[768, 2304] crc32=5e08bf63
PASS h.10.attn.c_attn.bias shape=[2304] crc32=7575b548
PASS h.10.attn.c_proj.weight shape=[768, 768] crc32=3c774cfe
PASS h.10.attn.c_proj.bias shape=[768] crc32=f254f310
PASS h.10.ln_2.weight shape=[768] crc32=8343c266
PASS h.10.ln_2.bias shape=[768] crc32=eac1e8d6
PASS h.10.mlp.c_fc.weight shape=[768, 3072] crc32=702857b0
PASS h.10.mlp.c_fc.bias shape=[3072] crc32=761e513e
PASS h.10.mlp.c_proj.weight shape=[3072, 768] crc32=d34573a8
PASS h.10.mlp.c_proj.bias shape=[768] crc32=4af7f176
PASS h.11.ln_1.weight shape=[768] crc32=d265810e
PASS h.11.ln_1.bias shape=[768] crc32=64a76217
PASS h.11.attn.c_attn.weight shape=[768, 2304] crc32=011a3c30
PASS h.11.attn.c_attn.bias shape=[2304] crc32=95e9105c
PASS h.11.attn.c_proj.weight shape=[768, 768] crc32=0693462e
PASS h.11.attn.c_proj.bias shape=[768] crc32=1ca96691
PASS h.11.ln_2.weight shape=[768] crc32=5894078f
PASS h.11.ln_2.bias shape=[768] crc32=3e3eca73
PASS h.11.mlp.c_fc.weight shape=[768, 3072] crc32=ad288899
PASS h.11.mlp.c_fc.bias shape=[3072] crc32=a858a7e8
PASS h.11.mlp.c_proj.weight shape=[3072, 768] crc32=b41639c2
PASS h.11.mlp.c_proj.bias shape=[768] crc32=d19c1af9
PASS ln_f.weight shape=[768] crc32=66271b90
PASS ln_f.bias shape=[768] crc32=71261dab
PASS all 148 tensor shapes, checksums, offsets and sizes

```

</details>
