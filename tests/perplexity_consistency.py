"""The C++ and PyTorch perplexity code paths must agree on the tiny model."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from python.export import make_tiny
from python.perplexity import score
from tests.native_process import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    ids = np.random.default_rng(42).integers(0, 97, 300).tolist()
    args.directory.mkdir(parents=True, exist_ok=True)
    ids_path = args.directory / "ids.txt"
    ids_path.write_text(" ".join(map(str, ids)), encoding="utf-8")
    torch.set_num_threads(1)
    model = make_tiny()

    def logits_fn(window_ids):
        with torch.inference_mode():
            return model(torch.tensor([window_ids]), use_cache=False).logits[0]

    for window, stride in [(64, 32), (64, 64), (50, 17)]:
        expected = score(logits_fn, ids, window, stride)
        for kernel, threads in [("reference", 1), ("avx2", 3)]:
            command = [
                str(args.executable.resolve()),
                str(args.model.resolve()),
                str(ids_path),
                str(window),
                str(stride),
                str(threads),
                kernel,
            ]
            result = capture(command)
            assert result.returncode == 0, result.stderr
            actual = json.loads(result.stdout)
            for key in ("tokens", "scored", "windows"):
                assert actual[key] == expected[key], (key, actual, expected)
            if stride < window:
                # Overlapping windows give every token after the first a prefix.
                assert actual["scored"] == len(ids) - 1, "every later token is scored once"
            relative = abs(actual["perplexity"] / expected["perplexity"] - 1)
            assert relative < 1e-5, (kernel, window, stride, actual, expected)
            print(
                f"PASS window {window} stride {stride} {kernel}: perplexity "
                f"{actual['perplexity']:.6f} vs PyTorch {expected['perplexity']:.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
