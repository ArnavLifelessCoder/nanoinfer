# M1 review packet

M1 contains the C++20 skeleton, GPT-2 exporter, aligned f32 format, read-only
Windows/POSIX mappings, validated zero-copy tensor views and the native inspector.
M2 has not started. Only branch `m1` is published for review.

The gating run is **ci on the final `m1` head**, visible in
[the M1 workflow runs](https://github.com/ArnavLifelessCoder/nanoinfer/actions/workflows/ci.yml?query=branch%3Am1).
Its commit must match `m1`, and all four platform/preset jobs plus lint must pass.
The exact final run link is supplied with the review handoff. The numeric run in
`results/m1-ci.json` is historical evidence, not the current gate.

The previous final review head was `dd3d5ba`, gated by
[run 39421570280](https://github.com/ArnavLifelessCoder/nanoinfer/actions/runs/39421570280),
not the earlier snapshot run. That gate is superseded by the UBSan enforcement
fix: Linux ASan now includes a regression that requires a deliberate signed
overflow to report a UBSan diagnostic and exit nonzero.

## Acceptance and full test output

| Criterion | Evidence |
| --- | --- |
| Every pretrained tensor shape and checksum agrees with the exporter | [Debug: all 148 tensors](../results/m1-debug-gpt2.txt), [ASan: all 148 tensors](../results/m1-asan-gpt2.txt) |
| Truncated and corrupted input gives clean errors | Unit tests cover every truncation boundary and byte corruption; CLI integration covers malformed files |
| Local Debug and ASan | [Complete Debug output](../results/m1-debug-tests.txt), [complete ASan output](../results/m1-asan-tests.txt): 12 C++ tests and 4 CTest entries per preset |
| Exporter correctness | [Full pytest output](../results/m1-python-tests.txt): 20 tests |
| Local native Release preset | [Complete output](../results/m1-release-tests.txt): 4 CTest entries |
| Windows/Ubuntu Debug and ASan plus lint | Final `m1` head's gating run above; Linux ASan additionally checks fatal UBSan behavior |
| Small golden set and repeatable export | [Artifact audit](../results/m1-summary.json), [repeated-export result](../results/m1-export-reproducibility.txt) |
| TSan | Not applicable until runtime threads exist; Ubuntu job is reserved for M5 |

Full configure/build logs are also in `results/`. No performance numbers are
claimed. Log path prefixes are replaced with portable placeholders, as described
in [results/README.md](../results/README.md).

## Changes from the first review

- `.github/workflows/ci.yml`: pytest uses the runner's existing temporary directory.
- `tests/test_loader.cpp`: bind malformed-metadata cases by reference for GCC's
  `-Werror=range-loop-construct`.
- `CMakeLists.txt`: select the project venv explicitly, since hosted runner
  `Python3_ROOT_DIR` can otherwise select system Python without dependencies.
- `CMakeLists.txt`, `tests/ubsan_probe.cpp`, `tests/check_ubsan_exit.cmake`:
  disable GCC/Clang sanitizer recovery in compilation and linking; verify a
  deliberate UBSan violation exits nonzero.
- `CMakeLists.txt`, `CMakePresets.json`, `README.md`: benchmark-only native Release
  flags; portable builds can disable `NANOINFER_NATIVE_RELEASE`.
- `.gitignore`: workstation-only spec and instruction exclusions moved to
  `.git/info/exclude`.
- `scripts/sanitize-results.ps1`, `scripts/test.ps1`, `results/*.txt`: normalize
  machine-specific path prefixes while retaining complete diagnostics.
- `tests/test_sanitize_results.ps1`: preserve HTTP(S) URLs while redacting drive
  paths, and verify repeated sanitization is idempotent; run in Windows Debug CI.
- `golden/README.md`: explicitly record the local PyTorch 2.11.0+cu128 build,
  executed entirely on CPU with one thread. CI uses the locked CPU wheel.
- `scripts/check_export_reproducibility.py`: reproduce the pinned model revision,
  compare model SHA-256, manifests and every golden array, and save the report.
- `scripts/verify_m1_artifacts.py`: include hosted CI evidence in the artifact audit.
- This review packet now uses relative repository links.

[CI regression summaries](../results/m1-ci-regressions.md) link to the complete
hosted failure logs. The four raw log dumps have been removed. The required
workflow tests remain enabled.

## Reproduction

From PowerShell at the repository root:

```powershell
. .\scripts\dev-shell.ps1
New-Item -ItemType Directory -Force build | Out-Null
python -m pytest --basetemp=build/pytest -v
.\scripts\test.ps1 debug
.\scripts\test.ps1 asan
python scripts/check_export_reproducibility.py
python scripts/verify_m1_artifacts.py
```

See [README](../README.md) for clean environment setup and
[golden/README](../golden/README.md) for the recorded local wheel distinction.
Weights are ignored; only the small golden fixtures are committed.

## Next milestone boundary

After M1 approval and publication to `main`, M2 will branch as `m2` from approved
`main`. It must validate expected tensor names and shapes against the header
configuration before implementing the reference forward pass. That semantic
validation is additional to M1's binary layout, checksum and bounds validation.
