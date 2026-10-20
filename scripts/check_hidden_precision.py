"""Diagnose the absolute hidden-state tolerance against a float64 PyTorch oracle."""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402


def main() -> None:
    torch.set_num_threads(1)
    metadata = json.loads((ROOT / "golden/manifest.json").read_text(encoding="utf-8"))
    model = (
        GPT2LMHeadModel.from_pretrained(
            "gpt2",
            revision=metadata["revision"],
            local_files_only=True,
            attn_implementation="eager",
            dtype=torch.float64,
        )
        .cpu()
        .eval()
    )
    results = []
    with torch.inference_mode():
        for prompt in metadata["prompts"]:
            captured = {}
            hooks = []
            for layer, block in enumerate(model.transformer.h):

                def capture(_module, _inputs, output, layer=layer):
                    captured[f"layer_{layer}"] = output[0][0].cpu().numpy().copy()

                hooks.append(block.register_forward_hook(capture))
            try:
                model(torch.tensor([prompt["ids"]]), use_cache=False)
            finally:
                for hook in hooks:
                    hook.remove()
            differences = {}
            with np.load(ROOT / "golden" / prompt["file"], allow_pickle=False) as golden:
                for key, oracle in captured.items():
                    rounded = oracle.astype(np.float32)
                    target = golden[key]
                    largest = float(np.max(np.abs(target)))
                    differences[key] = {
                        "max_abs_float64_rounded_vs_golden": float(
                            np.max(np.abs(rounded.astype(np.float64) - target))
                        ),
                        "largest_abs_golden": largest,
                        "float32_spacing_at_largest": float(np.spacing(np.float32(largest))),
                    }
            results.append(dict(file=prompt["file"], layers=differences))
    report = {
        "purpose": "diagnostic only; does not change runtime arithmetic or acceptance thresholds",
        "torch": torch.__version__,
        "model_revision": metadata["revision"],
        "oracle_dtype": "float64, rounded to float32 for comparison",
        "prompts": results,
    }
    text = json.dumps(report, indent=2) + "\n"
    (ROOT / "results/m2-hidden-precision-diagnostic.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
