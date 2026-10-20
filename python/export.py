"""Export GPT-2 f32 weights and independent PyTorch correctness fixtures."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

import numpy as np  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from python.model_format import write_model  # noqa: E402


def make_tiny(seed: int = 42) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    config = GPT2Config(
        n_layer=2,
        n_embd=16,
        n_head=2,
        n_positions=64,
        vocab_size=97,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
        bos_token_id=96,
        eos_token_id=96,
    )
    config._attn_implementation = "eager"
    return GPT2LMHeadModel(config).float().cpu().eval()


def export_weights(model: GPT2LMHeadModel, path: Path) -> dict:
    config = model.config
    if (
        config.activation_function != "gelu_new"
        or config.layer_norm_epsilon != 1e-5
        or not config.scale_attn_weights
        or config.scale_attn_by_inverse_layer_idx
        or config.reorder_and_upcast_attn
        or config.add_cross_attention
        or config.n_inner not in (None, 4 * config.n_embd)
    ):
        raise ValueError("unsupported GPT-2 configuration")
    if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
        raise ValueError("output embedding must be tied to wte")
    tensors = {}
    for name, parameter in model.transformer.named_parameters():
        # Conv1D already stores [in, out]. Keep this layout for x @ weight.
        tensors[name] = parameter.detach().cpu().float().numpy()
    manifest = write_model(
        path,
        dict(
            layers=config.n_layer,
            d_model=config.n_embd,
            heads=config.n_head,
            vocab=config.vocab_size,
            ctx=config.n_positions,
        ),
        tensors,
    )
    manifest["source"] = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "revision": getattr(config, "_commit_hash", None),
        "weight_layout": "Conv1D [in, out]; embeddings [vocab/ctx, d_model]",
    }
    return manifest


def write_golden(model: GPT2LMHeadModel, directory: Path, tiny: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    prompts = ["Hello", "The cat", "Why?"]
    ids = (
        [[1], [2, 7], [3, 4, 5]]
        if tiny
        else [tiktoken.get_encoding("gpt2").encode(prompt) for prompt in prompts]
    )
    metadata = {
        "model": "tiny seeded GPT-2" if tiny else "gpt2",
        "seed": 42,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "revision": getattr(model.config, "_commit_hash", None),
        "attention": "eager",
        "dtype": "float32",
        "threads": 1,
        "hidden_state_semantics": "block outputs after both residuals; final_ln separately",
        "prompts": [],
    }
    with torch.inference_mode():
        for index, tokens in enumerate(ids):
            arrays = {}
            hooks = []
            for layer, block in enumerate(model.transformer.h):

                def capture(_module, _inputs, output, layer=layer):
                    arrays[f"layer_{layer}"] = output[0][0].cpu().numpy().copy()

                hooks.append(block.register_forward_hook(capture))
            inputs = torch.tensor([tokens], dtype=torch.long)
            try:
                output = model(inputs, use_cache=False, output_hidden_states=True)
            finally:
                for hook in hooks:
                    hook.remove()
            arrays["input_ids"] = np.asarray(tokens, dtype="<i4")
            arrays["logits"] = output.logits[0].cpu().numpy()
            arrays["final_ln"] = output.hidden_states[-1][0].cpu().numpy()
            # No EOS early stop: the M2 contract compares exactly 32 new tokens.
            sequence = inputs
            generated = []
            for _ in range(32):
                token = model(sequence, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                generated.append(token.item())
                sequence = torch.cat((sequence, token), dim=1)
            arrays["greedy_ids"] = np.asarray(generated, dtype="<i4")
            filename = f"prompt_{index}.npz"
            np.savez_compressed(directory / filename, **arrays)
            metadata["prompts"].append(
                {"text": None if tiny else prompts[index], "ids": tokens, "file": filename}
            )
    (directory / "manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "gpt2.bin")
    parser.add_argument("--golden", type=Path, default=ROOT / "golden")
    parser.add_argument("--tiny", action="store_true", help="seeded test model, no download")
    parser.add_argument("--revision", default="main", help="Hugging Face model revision")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    if args.tiny:
        model = make_tiny()
    else:
        model = (
            GPT2LMHeadModel.from_pretrained(
                "gpt2",
                revision=args.revision,
                attn_implementation="eager",
                dtype=torch.float32,
            )
            .cpu()
            .eval()
        )
        expected = (12, 768, 12, 50257, 1024)
        actual = (
            model.config.n_layer,
            model.config.n_embd,
            model.config.n_head,
            model.config.vocab_size,
            model.config.n_positions,
        )
        if actual != expected:
            raise ValueError(f"expected GPT-2 small config {expected}; got {actual}")
    manifest = export_weights(model, args.output)
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    write_golden(model, args.golden, args.tiny)


if __name__ == "__main__":
    main()
