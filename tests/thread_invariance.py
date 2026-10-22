"""Require byte-identical forward captures and tokens for every thread count."""

import argparse
import json
from pathlib import Path

from tests.native_process import capture

PROMPTS = [[1], [2, 7], [3, 4, 5], list(range(1, 41))]


def run(executable, model, directory, ids, kernel, threads, timeout):
    output = directory / f"{kernel}-t{threads}" / f"p{len(ids)}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "ids.txt").write_text(" ".join(map(str, ids)), encoding="utf-8")
    result = capture(
        [
            str(executable),
            str(model),
            str(output / "ids.txt"),
            str(output),
            "16",
            kernel,
            str(threads),
        ],
        timeout=timeout,
    )
    assert result.returncode == 0, (kernel, threads, result.stderr)
    return {
        path.name: path.read_bytes()
        for path in sorted(output.iterdir())
        if path.suffix in (".f32", ".json")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--kernels", nargs="+", default=["packed", "avx2"])
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    executable, model = args.executable.resolve(), args.model.resolve()
    for kernel in args.kernels:
        for ids in PROMPTS:
            baseline = run(
                executable,
                model,
                args.directory.resolve(),
                ids,
                kernel,
                args.threads[0],
                args.timeout,
            )
            for threads in args.threads[1:]:
                other = run(
                    executable, model, args.directory.resolve(), ids, kernel, threads, args.timeout
                )
                assert other.keys() == baseline.keys()
                for name, content in baseline.items():
                    assert other[name] == content, (kernel, len(ids), threads, name)
            tokens = json.loads(baseline["output.json"])["greedy_ids"]
            print(
                f"PASS {kernel} prompt length {len(ids)}: identical captures and tokens "
                f"{tokens[:4]}... for threads {args.threads}",
                flush=True,
            )


if __name__ == "__main__":
    main()
