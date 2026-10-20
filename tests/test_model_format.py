import struct
import zlib

import numpy as np
import pytest

from python.model_format import ENTRY, HEADER, write_model

CONFIG = dict(layers=1, d_model=7, heads=1, vocab=11, ctx=8)


def test_binary_layout_and_checksums(tmp_path):
    path = tmp_path / "model.bin"
    arrays = {
        "matrix": np.arange(21, dtype=np.float32).reshape(7, 3),
        "bias": np.array([1.0, -2.0, 0.25], dtype=np.float32),
    }
    manifest = write_model(path, CONFIG, arrays)
    data = path.read_bytes()
    assert HEADER.size == 64
    assert ENTRY.size == 128
    assert data[:8] == b"GPT2RT01"
    assert struct.unpack_from("<I", data, 8)[0] == 1
    assert struct.unpack_from("<Q", data, 48)[0] == len(data)
    assert struct.unpack_from("<I", data, 56)[0] == zlib.crc32(data[64:320], zlib.crc32(data[:56]))
    for index, tensor in enumerate(manifest["tensors"]):
        entry = ENTRY.unpack_from(data, 64 + index * 128)
        assert entry[0].rstrip(b"\0").decode() == tensor["name"]
        assert list(entry[3 : 3 + entry[2]]) == tensor["shape"]
        offset, size, checksum = entry[7:10]
        assert offset % 64 == 0
        payload = data[offset : offset + size]
        assert payload == arrays[tensor["name"]].astype("<f4").tobytes()
        assert zlib.crc32(payload) == checksum == tensor["crc32"]


@pytest.mark.parametrize(
    "tensor",
    [
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((1,) * 5, dtype=np.float32),
        np.zeros(3, dtype=np.float64),
        np.array([np.inf], dtype=np.float32),
        np.array(1, dtype=np.float32),
    ],
)
def test_reject_invalid_tensors(tmp_path, tensor):
    with pytest.raises(ValueError):
        write_model(tmp_path / "bad.bin", CONFIG, {"bad": tensor})


@pytest.mark.parametrize("name", ["", "x" * 64, "embedded\0null", "\u03bb"])
def test_reject_invalid_names(tmp_path, name):
    with pytest.raises(ValueError):
        write_model(tmp_path / "bad.bin", CONFIG, {name: np.zeros(1, dtype=np.float32)})


@pytest.mark.parametrize(
    "config",
    [
        dict(CONFIG, heads=2),
        dict(CONFIG, ctx=0),
        dict(CONFIG, layers=-1),
        dict(CONFIG, vocab=2**32),
        dict(CONFIG, layers=1.5),
    ],
)
def test_reject_invalid_config(tmp_path, config):
    with pytest.raises(ValueError):
        write_model(tmp_path / "bad.bin", config, {"x": np.zeros(1, dtype=np.float32)})


def test_reject_empty_model(tmp_path):
    with pytest.raises(ValueError):
        write_model(tmp_path / "bad.bin", CONFIG, {})
