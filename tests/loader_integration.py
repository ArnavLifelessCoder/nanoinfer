"""Exercise the native loader with random and awkward Python-generated tensors."""

import argparse
import json
from pathlib import Path

import numpy as np

from python.model_format import write_model
from tests.native_process import capture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(f"Random seed: {args.seed}", flush=True)
    generator = np.random.default_rng(args.seed)
    tensors = {}
    for index, size in enumerate([1, 7, 63, 65, 767, 769]):
        shape = (size,) if index % 2 == 0 else (7, size)
        tensors[f"awkward_{size}"] = generator.standard_normal(shape).astype(np.float32)
    for index in range(25):
        shape = tuple(int(x) for x in generator.integers(1, 8, generator.integers(1, 5)))
        tensors[f"random_{index}"] = generator.standard_normal(shape).astype(np.float32)
    tensors['escaped_"\\\t'] = np.array([0.0, -0.0, 1e-30, 1e30], dtype=np.float32)
    model = args.directory / "weights \u03bb.bin"
    expected = write_model(model, dict(layers=1, d_model=7, heads=1, vocab=11, ctx=8), tensors)
    executable = str(args.inspector.resolve())
    command = [executable, str(model.resolve()), "--layout-only"]
    result = capture(command)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    print(f"PASS {len(tensors)} random/awkward tensors, all ranks, Unicode path and JSON escaping")
    valid = model.read_bytes()
    malformed = {
        "empty": b"",
        "header": valid[:20],
        "table": valid[:100],
        "payload": valid[:-100],
        "trailing": valid + b"x",
        "corrupt": valid[:-64] + bytes([valid[-64] ^ 1]) + valid[-63:],
    }
    for label, data in malformed.items():
        model.write_bytes(data)
        result = capture(command)
        assert result.returncode == 1, (label, result.returncode, result.stderr)
        assert result.stdout == ""
        assert result.stderr.startswith("model error:")
        print(f"PASS clean error for {label}: {result.stderr.strip()}")
    result = capture([executable])
    assert result.returncode == 2 and "usage:" in result.stderr
    print("PASS missing CLI argument")


if __name__ == "__main__":
    main()
