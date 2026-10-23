"""Binary writer for format versions 1 and 2. All fields and payloads are little endian.

Version 1 holds only f32 tensors. Version 2 adds dtype 2 (q8): symmetric int8
values in groups along one axis of a matrix, followed by their f32 scales. A
file is written as version 2 only when it contains a q8 tensor, so f32 exports
stay byte-identical to version 1.
"""

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = b"GPT2RT01"
HEADER = struct.Struct("<8s8I2Q2I")
ENTRY = struct.Struct("<64s2I4Q2Q2I")
CONFIG_KEYS = ("layers", "d_model", "heads", "vocab", "ctx")
DTYPE_F32, DTYPE_Q8 = 1, 2


@dataclass(frozen=True)
class Quantized:
    """int8 matrix q with float32 scales, grouped along `axis` (the input dimension)."""

    q: np.ndarray
    scales: np.ndarray
    axis: int
    group: int


def q8_payload(tensor: Quantized) -> bytes:
    """int8 values, zero padding to a 4-byte boundary, then the f32 scales."""
    q = np.ascontiguousarray(tensor.q, dtype=np.int8)
    scales = np.ascontiguousarray(tensor.scales, dtype="<f4")
    expected = list(q.shape)
    expected[tensor.axis] //= tensor.group
    if (
        q.ndim != 2
        or tensor.axis not in (0, 1)
        or not 0 < tensor.group < 65536
        or q.shape[tensor.axis] % tensor.group
        or list(scales.shape) != expected
    ):
        raise ValueError("q8 tensors are matrices with one scale per group")
    if not np.isfinite(scales).all() or (scales < 0).any():
        raise ValueError("q8 scales must be finite and nonnegative")
    values = q.tobytes()
    return values + bytes((-len(values)) % 4) + scales.tobytes()


def write_model(path: Path, config: dict, tensors: dict) -> dict:
    """Write canonical row-major tensors; return the independent round-trip manifest."""
    values = [config[key] for key in CONFIG_KEYS]
    if any(type(value) is not int or not 0 < value < 2**32 for value in values):
        raise ValueError("config values must be positive uint32 integers")
    if config["d_model"] % config["heads"]:
        raise ValueError("d_model must be divisible by heads")
    if not tensors:
        raise ValueError("a model needs tensors")
    version = 2 if any(isinstance(t, Quantized) for t in tensors.values()) else 1
    offset = HEADER.size + ENTRY.size * len(tensors)
    table = bytearray()
    payloads = []
    manifest = {"format_version": version, "config": dict(config), "tensors": []}
    for name, tensor in tensors.items():
        encoded = name.encode("ascii")
        if not encoded or len(encoded) >= 64 or b"\0" in encoded:
            raise ValueError("tensor names must contain 1..63 non-null ASCII bytes")
        if isinstance(tensor, Quantized):
            payload = q8_payload(tensor)
            shape = list(tensor.q.shape)
            dtype, extra, label = DTYPE_Q8, tensor.group | (tensor.axis << 16), "q8"
        else:
            array = np.asarray(tensor)
            if array.dtype.kind != "f" or array.dtype.itemsize != 4:
                raise ValueError("unquantized tensors must be float32")
            if not 1 <= array.ndim <= 4 or any(dim == 0 for dim in array.shape):
                raise ValueError("tensor rank must be 1..4 with nonzero dimensions")
            array = np.ascontiguousarray(array, dtype="<f4")
            if not np.isfinite(array).all():
                raise ValueError("weights must be finite")
            payload = array.tobytes()
            shape = list(array.shape)
            dtype, extra, label = DTYPE_F32, 0, "f32"
        checksum = zlib.crc32(payload)
        table.extend(
            ENTRY.pack(
                encoded,
                dtype,
                len(shape),
                *(shape + [0] * (4 - len(shape))),
                offset,
                len(payload),
                checksum,
                extra,
            )
        )
        entry = {
            "name": name,
            "dtype": label,
            "shape": shape,
            "offset": offset,
            "byte_size": len(payload),
            "crc32": checksum,
        }
        if label == "q8":
            entry.update(group=tensor.group, axis=tensor.axis)
        manifest["tensors"].append(entry)
        payloads.append(payload)
        offset = (offset + len(payload) + 63) // 64 * 64
    header = HEADER.pack(
        MAGIC, version, *values, len(tensors), ENTRY.size, HEADER.size + len(table), offset, 0, 0
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
