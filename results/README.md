# Validation evidence

M1 results are correctness checks, not performance benchmarks. Complete local
Debug, ASan and Release CTest output is in `m1-*-tests.txt`. The separate
`m1-debug-gpt2.txt` and `m1-asan-gpt2.txt` logs compare all pretrained tensors.

`scripts/test.ps1` records native configure, build and test output.
`scripts/check_export_reproducibility.py` produces the repeated-export report.
`scripts/verify_m1_artifacts.py` checks saved manifests, arrays and local logs.
The CI run metadata identifies the tested commit and links to all hosted jobs.

`scripts/sanitize-results.ps1` replaces machine-specific path prefixes with
`<project>`, `<user>`, `<runner>`, `<drive>`, `<hosted-toolcache>` and program-directory
placeholders. Logs retain the original diagnostics and test results; line
endings and trailing whitespace are normalized. Files with `regression`, `initial-failure` or `bootstrap`
in their names preserve earlier failures and preliminary validation.
