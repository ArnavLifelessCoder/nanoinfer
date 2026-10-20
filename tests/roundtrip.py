"""Compare C++ loader output with the Python exporter manifest, for every tensor."""

import argparse
import json
from pathlib import Path

if __package__:
    from .native_process import capture
else:
    from native_process import capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    completed = capture([str(args.inspector.resolve()), str(args.model.resolve())])
    completed.check_returncode()
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
