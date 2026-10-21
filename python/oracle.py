"""Write float64 PyTorch hidden states for the golden prompts.

The fp32 golden fixes one particular float32 summation order. An optimized
kernel that sums in a different order can be closer to the exact result yet
farther from that order, so hidden states are also judged against this
float64 oracle: a kernel must be no less accurate than PyTorch float32 itself.
"""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402

from python.export import make_tiny  # noqa: E402


def oracle_states(model, tokens):
    """Every block output and the final LayerNorm output, in float64."""
    arrays = {}
    hooks = []
    for layer, block in enumerate(model.transformer.h):

        def capture(_module, _inputs, output, layer=layer):
            arrays[f"layer_{layer}"] = output[0][0].cpu().numpy().copy()

        hooks.append(block.register_forward_hook(capture))
    try:
        with torch.inference_mode():
            output = model(
                torch.tensor([tokens], dtype=torch.long), use_cache=False, output_hidden_states=True
            )
    finally:
        for hook in hooks:
            hook.remove()
    arrays["final_ln"] = output.hidden_states[-1][0].cpu().numpy()
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=ROOT / "golden")
    parser.add_argument("--tiny", action="store_true", help="seeded tiny model, no download")
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads((args.golden / "manifest.json").read_text(encoding="utf-8"))
    if args.tiny:
        model = make_tiny().double().eval()
    else:
        model = (
            GPT2LMHeadModel.from_pretrained(
                "gpt2",
                revision=manifest["revision"],
                attn_implementation="eager",
                dtype=torch.float64,
            )
            .cpu()
            .eval()
        )
    for prompt in manifest["prompts"]:
        arrays = oracle_states(model, prompt["ids"])
        assert all(array.dtype == np.float64 for array in arrays.values())
        np.savez_compressed(args.golden / f"oracle_{prompt['file']}", **arrays)
        print(f"wrote float64 oracle for {prompt['file']}", flush=True)


if __name__ == "__main__":
    main()
