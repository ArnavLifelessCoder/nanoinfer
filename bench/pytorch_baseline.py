"""PyTorch CPU baseline with the runtime's prompts, thread count and phases; prints JSON.

Prefill runs the whole prompt once with the KV cache enabled (the runtime's
prefill also fills its cache and returns logits for every position). Decode
prefills the prompt, then generates greedily one token at a time from the
cache, timing each step. float32, torch.inference_mode(), eager attention,
torch.set_num_threads(THREADS). One untimed warm-up precedes each measurement.
"""

import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import torch  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", type=Path, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--prefill", type=int, nargs="+", default=[128, 512])
    parser.add_argument("--decode-prompt", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=128)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = json.loads((ROOT / "golden" / "manifest.json").read_text(encoding="utf-8"))
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", revision=manifest["revision"], attn_implementation="eager", dtype=torch.float32
    ).eval()
    ids = [int(token) for token in args.ids.read_text(encoding="utf-8").split()]
    report = dict(threads=torch.get_num_threads(), torch=torch.__version__, prefill=[])
    with torch.inference_mode():
        for length in args.prefill:
            prompt = torch.tensor([ids[:length]])
            model(prompt, use_cache=True)  # Warm-up.
            seconds = []
            for _ in range(args.repetitions):
                start = time.perf_counter()
                model(prompt, use_cache=True)
                seconds.append(time.perf_counter() - start)
            report["prefill"].append(dict(tokens=length, seconds=seconds))
        runs = []
        prompt = torch.tensor([ids[: args.decode_prompt]])
        for run in range(args.repetitions + 1):  # Run 0 is the warm-up.
            output = model(prompt, use_cache=True)
            past, token = output.past_key_values, output.logits[:, -1:].argmax(-1)
            steps = []
            for _ in range(args.decode_tokens):
                start = time.perf_counter()
                output = model(token, past_key_values=past, use_cache=True)
                steps.append(time.perf_counter() - start)
                past, token = output.past_key_values, output.logits[:, -1:].argmax(-1)
            if run:
                runs.append(steps)
        report["decode"] = dict(
            prompt_tokens=args.decode_prompt, new_tokens=args.decode_tokens, runs=runs
        )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
