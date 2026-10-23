"""Exercise the native loader on version 2 files with q8 tensors."""

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np

from python.model_format import ENTRY, HEADER, Quantized, write_model
from python.quantize import quantize
from tests.native_process import capture


def repair_metadata(data: bytearray) -> bytearray:
    """Recompute the metadata checksum so a test reaches the validation it targets."""
    count = struct.unpack_from("<I", data, 32)[0]
    table = bytes(data[HEADER.size : HEADER.size + count * ENTRY.size])
    struct.pack_into("<I", data, 56, zlib.crc32(table, zlib.crc32(bytes(data[:56]))))
    return data


def entry_offset(index: int) -> int:
    return HEADER.size + index * ENTRY.size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((64, 7)).astype(np.float32)
    embedding = rng.standard_normal((5, 96)).astype(np.float32)
    q0, s0 = quantize(weights, 0, 32)
    q1, s1 = quantize(embedding, 1, 32)
    tensors = {
        "conv": Quantized(q0, s0, 0, 32),
        "bias": rng.standard_normal(7).astype(np.float32),
        "embedding": Quantized(q1, s1, 1, 32),
    }
    path = args.directory / "quantized.bin"
    expected = write_model(path, dict(layers=1, d_model=7, heads=1, vocab=11, ctx=8), tensors)
    assert expected["format_version"] == 2
    inspector = str(args.inspector.resolve())
    result = capture([inspector, str(path.resolve()), "--layout-only"])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected
    print("PASS q8 tensors round-trip: dtype, shape, group, axis, offset, size, crc32")

    valid = bytearray(path.read_bytes())
    malformed = {}
    downgraded = bytearray(valid)
    struct.pack_into("<I", downgraded, 8, 1)
    malformed["q8 in a version 1 file"] = repair_metadata(downgraded)
    for label, extra in [("group 0", 0), ("axis 2", 32 | 2 << 16), ("group not dividing", 24)]:
        changed = bytearray(valid)
        struct.pack_into("<I", changed, entry_offset(0) + 124, extra)
        malformed[label] = repair_metadata(changed)
    resized = bytearray(valid)
    size = struct.unpack_from("<Q", resized, entry_offset(0) + 112)[0]
    struct.pack_into("<Q", resized, entry_offset(0) + 112, size - 4)
    malformed["q8 size without its scales"] = repair_metadata(resized)
    corrupt = bytearray(valid)
    offset = struct.unpack_from("<Q", corrupt, entry_offset(0) + 104)[0]
    corrupt[offset] ^= 1
    malformed["corrupt q8 payload"] = corrupt
    malformed["truncated"] = valid[:-80]
    for label, data in malformed.items():
        path.write_bytes(bytes(data))
        result = capture([inspector, str(path.resolve()), "--layout-only"])
        assert result.returncode == 1 and not result.stdout, (label, result)
        assert result.stderr.startswith("model error:"), (label, result.stderr)
        print(f"PASS clean error for {label}: {result.stderr.strip()}")


if __name__ == "__main__":
    main()
