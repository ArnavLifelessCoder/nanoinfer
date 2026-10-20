# Validation evidence

M1 results are correctness checks, not performance benchmarks. Complete local
Debug, ASan and Release CTest output is in `m1-*-tests.txt`. The separate
`m1-debug-gpt2.txt` and `m1-asan-gpt2.txt` logs compare all pretrained tensors.

`scripts/test.ps1` records native configure, build and test output.
`scripts/check_export_reproducibility.py` produces the repeated-export report.
`scripts/verify_m1_artifacts.py` checks saved manifests, arrays and local logs.
The CI run metadata is a historical snapshot and identifies its tested commit.
The review packet identifies the gate for the current review.

`scripts/sanitize-results.ps1` replaces machine-specific path prefixes with
`<project>`, `<user>`, `<runner>`, `<drive>`, `<hosted-toolcache>` and program-directory
placeholders. Logs retain the original diagnostics and test results; line
endings and trailing whitespace are normalized. Local regression and bootstrap
files preserve earlier validation. [CI regression history](m1-ci-regressions.md)
summarizes the hosted failures with links to their complete logs.
