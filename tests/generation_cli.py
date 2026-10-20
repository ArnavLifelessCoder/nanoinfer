"""Exercise cached generation, seeded sampling, Unicode paths and CLI failures."""

import argparse
import json
from pathlib import Path

import numpy as np

from tests.native_process import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve() / "generation \u03bb"
    directory.mkdir(parents=True, exist_ok=True)
    ids_path = directory / "prompt \u03bb.txt"
    base = [
        str(args.executable.resolve()),
        "--model",
        str(args.model.resolve()),
        "--prompt-ids",
        str(ids_path),
    ]

    def run(ids, options, success=True):
        ids_path.write_text(" ".join(map(str, ids)), encoding="utf-8")
        result = capture(base + options)
        assert result.returncode == (0 if success else 1), result
        if success:
            assert not result.stderr, result.stderr
            output = json.loads(result.stdout)
            assert output["prompt_ids"] == ids
            return output["generated_ids"]
        assert not result.stdout and "generation error:" in result.stderr, result

    manifest = json.loads((args.golden / "manifest.json").read_text(encoding="utf-8"))
    for prompt in manifest["prompts"]:
        with np.load(args.golden / prompt["file"]) as golden:
            assert run(prompt["ids"], ["--max-new", "32"]) == golden["greedy_ids"].tolist()
    print("PASS cached greedy matches all three PyTorch golden continuations", flush=True)
    options = [
        "--max-new",
        "32",
        "--temp",
        "0.8",
        "--top-k",
        "30",
        "--top-p",
        "0.9",
        "--seed",
        "1234",
    ]
    first = run([2, 7], options)
    assert first == run([2, 7], options)
    assert first != run([2, 7], options[:-1] + ["5678"])
    assert len(first) == 32 and all(0 <= token < 97 for token in first)
    assert run([2, 7], ["--max-new", "32", "--temp", "1", "--top-k", "1"]) == run(
        [2, 7], ["--max-new", "32"]
    )
    print("PASS seeded replay, different seeds and top-k=1 greedy equivalence", flush=True)
    assert run([1] * 64, ["--max-new", "0"]) == []
    assert len(run([1] * 63, ["--max-new", "1"])) == 1
    for ids, options in [
        ([], ["--max-new", "0"]),
        ([1] * 64, ["--max-new", "1"]),
        ([1] * 65, ["--max-new", "0"]),
        ([97], ["--max-new", "0"]),
        ([-1], ["--max-new", "0"]),
        (["1x"], ["--max-new", "0"]),
        ([4294967296], ["--max-new", "0"]),
        ([1], ["--max-new", "-1"]),
        ([1], ["--max-new", "18446744073709551616"]),
        ([1], ["--temp", "nan"]),
        ([1], ["--temp", "inf"]),
        ([1], ["--temp", "-1"]),
        ([1], ["--top-p", "0"]),
        ([1], ["--top-p", "1.1"]),
        ([1], ["--top-k", "-1"]),
        ([1], ["--seed", "-1"]),
        ([1], ["--wat", "2"]),
        ([1], ["--max-new"]),
        ([1], ["--max-new", "0", "--max-new", "0"]),
    ]:
        run(ids, options, success=False)
    missing = capture([base[0]])
    assert missing.returncode == 1 and not missing.stdout
    help_result = capture([base[0], "--help"])
    assert help_result.returncode == 0 and "stdout: JSON" in help_result.stdout
    print("PASS context boundaries, numeric validation, option errors and help", flush=True)


if __name__ == "__main__":
    main()
