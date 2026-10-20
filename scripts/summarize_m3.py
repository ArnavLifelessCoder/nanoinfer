"""Summarize the saved Debug and ASan cache acceptance reports."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    summaries = {}
    for preset in ("debug", "asan"):
        for model in ("tiny", "pretrained"):
            path = root / "results" / f"m3-{preset}-{model}.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            assert report["passed"]
            prompts = report["prompts"]
            assert len(prompts) == 3
            assert all(p["greedy_match"] and p["greedy_tokens"] == 32 for p in prompts)
            positions = [row for p in prompts for row in p["cache_vs_full_recompute"]["positions"]]
            assert all(p["cache_vs_full_recompute"]["passed"] for p in prompts)
            summaries[f"{preset}_{model}"] = dict(
                source=path.relative_to(root).as_posix(),
                prompt_count=len(prompts),
                greedy_tokens_per_prompt=32,
                checked_positions=len(positions),
                cache_max_abs_logit_diff=max(row["max_abs_logit_diff"] for row in positions),
                cache_max_abs_hidden_diff=max(row["max_abs_hidden_diff"] for row in positions),
                torch_max_abs_logit_diff=max(p["max_abs_diff"]["logits"] for p in prompts),
                torch_max_hidden_tolerance_ratio=max(
                    ratio for p in prompts for ratio in p["max_hidden_tolerance_ratio"].values()
                ),
            )
            assert summaries[f"{preset}_{model}"]["cache_max_abs_logit_diff"] == 0
            assert summaries[f"{preset}_{model}"]["cache_max_abs_hidden_diff"] == 0
    for model in ("tiny", "pretrained"):
        debug = {k: v for k, v in summaries[f"debug_{model}"].items() if k != "source"}
        asan = {k: v for k, v in summaries[f"asan_{model}"].items() if k != "source"}
        assert debug == asan, f"{model} Debug and ASan summaries differ"
    output = root / "results" / "m3-summary.json"
    output.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(output.relative_to(root))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
