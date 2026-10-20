"""Validate saved M2 acceptance reports and summarize their measured errors."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summarize():
    manifest = json.loads((ROOT / "golden/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["prompts"]) == 3
    results = {}
    for preset in ("debug", "asan"):
        path = ROOT / f"results/m2-{preset}-pretrained.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        assert report["passed"] and report["model"] == manifest["model"]
        assert report["tolerance"] == {
            "logits_max_abs_exclusive": 1e-3,
            "hidden_atol": 1e-4,
            "hidden_rtol": 1e-6,
        }
        assert [p["file"] for p in report["prompts"]] == [p["file"] for p in manifest["prompts"]]
        prompts = []
        for prompt in report["prompts"]:
            assert prompt["greedy_tokens"] == 32 and prompt["greedy_match"]
            assert not prompt["failures"]
            assert prompt["max_abs_diff"]["logits"] < 1e-3
            assert max(prompt["max_hidden_tolerance_ratio"].values()) <= 1
            assert set(prompt["max_hidden_tolerance_ratio"]) == {
                "final_ln",
                *(f"layer_{i}" for i in range(12)),
            }
            stem = Path(prompt["file"]).stem
            output = ROOT / f"build/{preset}/pretrained/{stem}/output.json"
            actual = json.loads(output.read_text(encoding="utf-8"))["greedy_ids"]
            with np.load(ROOT / "golden" / prompt["file"], allow_pickle=False) as golden:
                expected = golden["greedy_ids"].tolist()
            assert len(actual) == 32 and actual == expected
            prompts.append(
                dict(
                    file=prompt["file"],
                    logits_max_abs=prompt["max_abs_diff"]["logits"],
                    hidden_max_abs=max(
                        value for key, value in prompt["max_abs_diff"].items() if key != "logits"
                    ),
                    hidden_max_tolerance_ratio=max(prompt["max_hidden_tolerance_ratio"].values()),
                    greedy_ids=actual,
                    greedy_match=True,
                )
            )
        executable = ROOT / f"build/{preset}/gpt2ref.exe"
        if not executable.exists():
            executable = executable.with_suffix("")
        results[preset] = dict(
            report=str(path.relative_to(ROOT)).replace("\\", "/"),
            executable_sha256=digest(executable),
            prompts=prompts,
        )
    summary = dict(
        purpose="M2 correctness acceptance; no performance measurements",
        model_revision=manifest["revision"],
        model_sha256=digest(ROOT / "models/gpt2.bin"),
        golden_sha256={p["file"]: digest(ROOT / "golden" / p["file"]) for p in manifest["prompts"]},
        reference_torch=manifest["torch"],
        runs=results,
        passed=True,
    )
    (ROOT / "results/m2-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for preset, result in results.items():
        print(
            f"{preset}: logits max abs={max(p['logits_max_abs'] for p in result['prompts']):.12g}, "
            f"hidden max tolerance ratio="
            f"{max(p['hidden_max_tolerance_ratio'] for p in result['prompts']):.12g}, "
            "all 3 prompts match 32/32 greedy IDs"
        )


if __name__ == "__main__":
    summarize()
