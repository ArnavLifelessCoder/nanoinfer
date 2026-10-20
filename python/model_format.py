"""Version 1 binary writer. All fields and f32 payloads are little endian."""

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"GPT2RT01"
HEADER = struct.Struct("<8s8I2Q2I")
ENTRY = struct.Struct("<64s2I4Q2Q2I")
CONFIG_KEYS = ("layers", "d_model", "heads", "vocab", "ctx")


def write_model(path: Path, config: dict, tensors: dict) -> dict:
    """Write canonical row-major tensors; return the independent round-trip manifest."""
    values = [config[key] for key in CONFIG_KEYS]
    if any(type(value) is not int or not 0 < value < 2**32 for value in values):
        raise ValueError("config values must be positive uint32 integers")
    if config["d_model"] % config["heads"]:
        raise ValueError("d_model must be divisible by heads")
    if not tensors:
        raise ValueError("a model needs tensors")
    offset = HEADER.size + ENTRY.size * len(tensors)
    table = bytearray()
    payloads = []
    manifest = {"format_version": 1, "config": dict(config), "tensors": []}
    for name, tensor in tensors.items():
        encoded = name.encode("ascii")
        if not encoded or len(encoded) >= 64 or b"\0" in encoded:
            raise ValueError("tensor names must contain 1..63 non-null ASCII bytes")
        array = np.asarray(tensor)
        if array.dtype.kind != "f" or array.dtype.itemsize != 4:
            raise ValueError("version 1 supports only float32")
        if not 1 <= array.ndim <= 4 or any(dim == 0 for dim in array.shape):
            raise ValueError("tensor rank must be 1..4 with nonzero dimensions")
        array = np.ascontiguousarray(array, dtype="<f4")
        if not np.isfinite(array).all():
            raise ValueError("weights must be finite")
        payload = array.tobytes()
        checksum = zlib.crc32(payload)
        shape = list(array.shape)
        table.extend(
            ENTRY.pack(
                encoded,
                1,
                len(shape),
                *(shape + [0] * (4 - len(shape))),
                offset,
                len(payload),
                checksum,
                0,
            )
        )
        manifest["tensors"].append(
            {
                "name": name,
                "dtype": "f32",
                "shape": shape,
                "offset": offset,
                "byte_size": len(payload),
                "crc32": checksum,
            }
        )
        payloads.append(payload)
        offset = (offset + len(payload) + 63) // 64 * 64
    header = HEADER.pack(
        MAGIC, 1, *values, len(tensors), ENTRY.size, HEADER.size + len(table), offset, 0, 0
    )
    metadata_crc = zlib.crc32(table, zlib.crc32(header[:56]))
    header = header[:56] + struct.pack("<II", metadata_crc, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(table)
        for payload in payloads:
            stream.write(payload)
            stream.write(bytes((-len(payload)) % 64))
    return manifest
