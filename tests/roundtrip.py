"""Compare C++ loader output with the Python exporter manifest, for every tensor."""

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [str(args.inspector.resolve()), str(args.model.resolve())],
        capture_output=True,
        text=True,
        check=True,
    )
    actual = json.loads(completed.stdout)
    assert actual["format_version"] == expected["format_version"]
    assert actual["config"] == expected["config"]
    assert len(actual["tensors"]) == len(expected["tensors"])
    for loaded, exported in zip(actual["tensors"], expected["tensors"], strict=True):
        assert loaded == exported, (loaded, exported)
        print(f"PASS {loaded['name']} shape={loaded['shape']} crc32={loaded['crc32']:08x}")
    print(f"PASS all {len(actual['tensors'])} tensor shapes, checksums, offsets and sizes")


if __name__ == "__main__":
    main()
