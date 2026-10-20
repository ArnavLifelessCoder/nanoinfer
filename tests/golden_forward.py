"""Compare native per-layer states, every prompt logit and 32 greedy IDs."""

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def compare_array(actual, target, *, hidden):
    """Return max absolute error, elementwise tolerance ratio and pass status."""
    assert actual.shape == target.shape, (actual.shape, target.shape)
    assert actual.size > 0
    assert np.isfinite(actual).all() and np.isfinite(target).all()
    absolute = np.abs(actual.astype(np.float64) - target.astype(np.float64))
    difference = float(np.max(absolute))
    if not hidden:
        return difference, None, difference < 1e-3
    # Owner-approved M2 clarification: relative tolerance applies per element.
    allowed = 1e-4 + 1e-6 * np.abs(target.astype(np.float64))
    ratio = float(np.max(absolute / allowed))
    return difference, ratio, ratio <= 1


def check_prompt(executable, model, golden, directory, prompt, new_tokens):
    arrays_path = golden / prompt["file"]
    output = directory / arrays_path.stem
    output.mkdir(parents=True, exist_ok=True)
    ids = output / "ids.txt"
    ids.write_text(" ".join(str(token) for token in prompt["ids"]), encoding="utf-8")
    subprocess.run(
        [str(executable), str(model), str(ids), str(output), str(new_tokens)], check=True
    )
    metadata = json.loads((output / "output.json").read_text(encoding="utf-8"))
    assert metadata["positions"] == len(prompt["ids"])
    differences = {}
    scaled_errors = {}
    failures = []
    with np.load(arrays_path, allow_pickle=False) as expected:
        keys = ["logits", "final_ln"] + [f"layer_{i}" for i in range(metadata["layers"])]
        for key in keys:
            actual = np.fromfile(output / f"{key}.f32", dtype="<f4")
            target = expected[key]
            assert actual.size == target.size, (key, actual.size, target.shape)
            actual = actual.reshape(target.shape)
            difference, ratio, passed = compare_array(actual, target, hidden=key != "logits")
            differences[key] = difference
            if key == "logits":
                if not passed:
                    failures.append(f"logits: max abs diff {difference} >= 1e-3")
            else:
                scaled_errors[key] = ratio
                if not passed:
                    failures.append(f"{key}: max elementwise tolerance ratio {ratio}")
        greedy_match = np.array_equal(metadata["greedy_ids"], expected["greedy_ids"][:new_tokens])
        if not greedy_match:
            failures.append("greedy IDs differ")
    status = "FAIL" if failures else "PASS"
    print(f"{status} {prompt['file']}: {differences}; greedy_match={greedy_match}", flush=True)
    return dict(
        file=prompt["file"],
        max_abs_diff=differences,
        max_hidden_tolerance_ratio=scaled_errors,
        greedy_tokens=new_tokens,
        greedy_match=greedy_match,
        failures=failures,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--new-tokens", type=int, default=32, choices=range(33))
    parser.add_argument("--jobs", type=int, default=1, choices=range(1, 4))
    args = parser.parse_args()
    metadata = json.loads((args.golden / "manifest.json").read_text(encoding="utf-8"))

    def check(prompt):
        return check_prompt(
            args.executable.resolve(),
            args.model.resolve(),
            args.golden.resolve(),
            args.directory.resolve(),
            prompt,
            args.new_tokens,
        )

    # Independent correctness cases may run in separate processes. Each native
    # runtime is single-threaded; this is not a runtime throughput benchmark.
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(check, metadata["prompts"]))
    report = dict(
        model=metadata["model"],
        execution="scalar, single native thread, full-prefix recomputation",
        reference_torch=metadata["torch"],
        tolerance=dict(logits_max_abs_exclusive=1e-3, hidden_atol=1e-4, hidden_rtol=1e-6),
        prompts=results,
        passed=all(not prompt["failures"] for prompt in results),
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert report["passed"], "golden comparison failed; see per-prompt differences"
    print("PASS all golden logits, hidden states and greedy continuations", flush=True)


if __name__ == "__main__":
    main()
