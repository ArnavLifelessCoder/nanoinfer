"""Require complete CLI output across repeated short process lifetimes."""

import argparse
import json
from pathlib import Path

from tests.native_process import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    for index in range(args.count):
        if index % 25 == 0:
            print(f"Checking output pair {index + 1}/{args.count}", flush=True)
        inspected = capture([str(args.inspector.resolve()), str(args.model.resolve())])
        assert inspected.returncode == 0 and not inspected.stderr, (index, inspected)
        assert inspected.stdout, (index, inspected)
        manifest = json.loads(inspected.stdout)
        assert manifest["config"]["vocab"] == 97 and manifest["tensors"], index
        help_text = capture([str(args.generator.resolve()), "--help"])
        assert help_text.returncode == 0 and not help_text.stderr, (index, help_text)
        assert help_text.stdout.endswith("temp=0 selects greedy.\n"), (index, help_text)
    print(f"PASS complete JSON and help output in {args.count * 2} launches", flush=True)


if __name__ == "__main__":
    main()
