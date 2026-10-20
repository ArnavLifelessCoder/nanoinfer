"""Check cached golden outputs and exact full-recompute equivalence at every position."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.golden_forward import check_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, choices=range(1, 4))
    args = parser.parse_args()
    manifest = json.loads((args.golden / "manifest.json").read_text(encoding="utf-8"))

    def check(prompt):
        result = check_prompt(
            args.executable.resolve(),
            args.model.resolve(),
            args.golden.resolve(),
            args.directory.resolve(),
            prompt,
            32,
        )
        path = args.directory / Path(prompt["file"]).stem / "cache-check.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        positions = cache["positions"]
        assert cache["passed"]
        assert [row["absolute_position"] for row in positions] == list(
            range(len(prompt["ids"]) + 32)
        )
        assert all(
            row["max_abs_logit_diff"] == row["max_abs_hidden_diff"] == 0 for row in positions
        )
        assert [row["phase"] for row in positions] == (
            ["prefill"] * len(prompt["ids"]) + ["decode"] * 32
        )
        result["cache_vs_full_recompute"] = cache
        return result

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(check, manifest["prompts"]))
    report = dict(
        model=manifest["model"],
        reference_torch=manifest["torch"],
        execution="scalar cache vs scalar full recomputation; one native thread per process",
        tolerance=dict(logits_max_abs_exclusive=1e-3, hidden_atol=1e-4, hidden_rtol=1e-6),
        prompts=results,
        passed=all(not item["failures"] for item in results),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert report["passed"], "cached golden comparison failed"
    print("PASS cached PyTorch golden checks and exact cache/full-prefix equivalence", flush=True)


if __name__ == "__main__":
    main()
