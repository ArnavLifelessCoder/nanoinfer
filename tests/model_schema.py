"""Reject structurally valid files whose tensors disagree with GPT-2 config."""

import argparse
from pathlib import Path

import numpy as np

from python.model_format import write_model
from tests.native_process import capture


def tensor_shapes(config: dict) -> dict:
    d = config["d_model"]
    shapes = {
        "wte.weight": (config["vocab"], d),
        "wpe.weight": (config["ctx"], d),
        "ln_f.weight": (d,),
        "ln_f.bias": (d,),
    }
    for layer in range(config["layers"]):
        prefix = f"h.{layer}."
        for norm in ("ln_1", "ln_2"):
            shapes[prefix + norm + ".weight"] = (d,)
            shapes[prefix + norm + ".bias"] = (d,)
        for name, shape in {
            "attn.c_attn": (d, 3 * d),
            "attn.c_proj": (d, d),
            "mlp.c_fc": (d, 4 * d),
            "mlp.c_proj": (4 * d, d),
        }.items():
            shapes[prefix + name + ".weight"] = shape
            shapes[prefix + name + ".bias"] = (shape[1],)
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspector", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    config = dict(layers=2, d_model=7, heads=1, vocab=11, ctx=8)
    tensors = {name: np.zeros(shape, np.float32) for name, shape in tensor_shapes(config).items()}
    path = args.directory / "schema.bin"
    executable = str(args.inspector.resolve())

    def check(label, values, expected, header=config):
        write_model(path, header, values)
        result = capture([executable, str(path.resolve())])
        assert result.returncode == expected, (label, result.returncode, result.stderr)
        if expected:
            assert result.stdout == "" and result.stderr.startswith("model error:")
        print(f"PASS {label}", flush=True)

    check("valid complete model with awkward width", tensors, 0)
    for name, values in tensors.items():
        wrong = tensors.copy()
        wrong[name] = values.reshape(-1) if values.ndim == 2 else values.reshape(1, -1)
        check(f"reject wrong rank/shape: {name}", wrong, 1)
    wrong = tensors.copy()
    wrong["wte.weight"] = tensors["wte.weight"].T.copy()
    check("reject transposed embedding with same element count", wrong, 1)
    for name in tensors:
        check(f"reject missing tensor: {name}", {k: v for k, v in tensors.items() if k != name}, 1)
    check("reject untied extra lm_head", tensors | {"lm_head.weight": tensors["wte.weight"]}, 1)
    wrong = tensors.copy()
    wrong["h.2.ln_1.bias"] = wrong.pop("h.1.ln_1.bias")
    check("reject out-of-range layer index", wrong, 1)
    check("reject config width mismatch", tensors, 1, config | {"d_model": 8})
    check("reject huge header layer count", tensors, 1, config | {"layers": 0xFFFFFFFF})


if __name__ == "__main__":
    main()
