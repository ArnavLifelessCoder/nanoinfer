"""Compare every optimized GEMM forward with PyTorch and with the scalar reference.

Logits must stay within 1e-3 of PyTorch float32 and greedy tokens must match
exactly. A hidden state passes the approved fp32 bound, or, when a different
summation order moves it outside that bound, it must be at least as accurate
as PyTorch float32 against the float64 oracle (python/oracle.py), within the
approved 1e-4 absolute slack. Owner decision for M4.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tests.golden_forward import check_prompt, compare_array

ATOL = 1e-4


def oracle_error(path, oracle):
    actual = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(oracle.shape)
    return float(np.max(np.abs(actual - oracle)))


def judge(kernel_dir, reference_dir, golden, oracle, layers):
    """Per-tensor verdicts against PyTorch, the scalar reference and float64."""
    rows, failures = {}, []
    for key in ["logits", "final_ln"] + [f"layer_{i}" for i in range(layers)]:
        actual = np.fromfile(kernel_dir / f"{key}.f32", dtype="<f4").reshape(golden[key].shape)
        reference = np.fromfile(reference_dir / f"{key}.f32", dtype="<f4").reshape(actual.shape)
        torch_diff, torch_ratio, torch_ok = compare_array(
            actual, golden[key], hidden=key != "logits"
        )
        ref_diff, ref_ratio, ref_ok = compare_array(actual, reference, hidden=key != "logits")
        row = dict(
            max_abs_diff_vs_torch=torch_diff,
            tolerance_ratio_vs_torch=torch_ratio,
            max_abs_diff_vs_reference=ref_diff,
            tolerance_ratio_vs_reference=ref_ratio,
        )
        if key == "logits":
            if not torch_ok:
                failures.append(f"logits differ from PyTorch by {torch_diff} >= 1e-3")
            if not ref_ok:
                failures.append(f"logits differ from the scalar reference by {ref_diff} >= 1e-3")
        else:
            truth = oracle[key]
            kernel_error = oracle_error(kernel_dir / f"{key}.f32", truth)
            torch_error = float(np.max(np.abs(golden[key].astype(np.float64) - truth)))
            reference_error = oracle_error(reference_dir / f"{key}.f32", truth)
            row.update(
                max_abs_error_vs_float64=kernel_error,
                torch_float32_error_vs_float64=torch_error,
                reference_error_vs_float64=reference_error,
            )
            if not torch_ok and kernel_error > torch_error + ATOL:
                failures.append(f"{key}: outside the fp32 bound and less accurate than PyTorch")
            if not ref_ok and kernel_error > max(reference_error, torch_error) + ATOL:
                failures.append(f"{key}: outside the reference bound and less accurate than it")
        rows[key] = row
    return rows, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-executable", type=Path, required=True)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kernels", nargs="+", required=True)
    parser.add_argument("--timeout", type=float, default=60, help="seconds per native run")
    parser.add_argument("--extra", nargs="*", default=[], help="arguments after the kernel name")
    args = parser.parse_args()
    golden_dir = args.golden.resolve()
    manifest = json.loads((golden_dir / "manifest.json").read_text(encoding="utf-8"))
    directory = args.directory.resolve()
    model = args.model.resolve()
    results, passed = {}, True
    for prompt in manifest["prompts"]:
        stem = Path(prompt["file"]).stem
        reference = check_prompt(
            args.reference_executable.resolve(),
            model,
            golden_dir,
            directory / "reference",
            prompt,
            32,
            timeout=args.timeout,
        )
        passed &= reference["greedy_match"]
        results.setdefault("reference", []).append(reference)
        with (
            np.load(golden_dir / prompt["file"], allow_pickle=False) as golden,
            np.load(golden_dir / f"oracle_{prompt['file']}", allow_pickle=False) as oracle,
        ):
            golden_arrays = {key: golden[key] for key in golden.files}
            oracle_arrays = {key: oracle[key] for key in oracle.files}
        for kernel in args.kernels:
            run = check_prompt(
                args.executable.resolve(),
                model,
                golden_dir,
                directory / kernel,
                prompt,
                32,
                timeout=args.timeout,
                arguments=[kernel, *args.extra],
            )
            output = directory / kernel / stem
            layers = json.loads((output / "output.json").read_text(encoding="utf-8"))["layers"]
            rows, failures = judge(
                output, directory / "reference" / stem, golden_arrays, oracle_arrays, layers
            )
            if not run["greedy_match"]:
                failures.append("greedy IDs differ from PyTorch")
            result = dict(
                file=prompt["file"],
                greedy_match=run["greedy_match"],
                tensors=rows,
                failures=failures,
            )
            print(
                f"{'FAIL' if failures else 'PASS'} {kernel} {prompt['file']} {failures}", flush=True
            )
            passed &= not failures
            results.setdefault(kernel, []).append(result)
    report = dict(
        model=manifest["model"],
        reference_torch=manifest["torch"],
        rule=__doc__.strip(),
        tolerance=dict(logits_max_abs_exclusive=1e-3, hidden_atol=ATOL, hidden_rtol=1e-6),
        kernels=args.kernels,
        extra_arguments=args.extra,
        results=results,
        passed=passed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert passed, "an optimized forward failed; see the report"
    print(
        f"PASS {', '.join(args.kernels)} against PyTorch, float64 and the scalar reference",
        flush=True,
    )


if __name__ == "__main__":
    main()
