"""WikiText-2 test slice: download, tokenize, and PyTorch perplexity.

The slice is the first TOKENS tokens of the WikiText-2 raw test split
("\\n\\n".join of its rows, as in the Hugging Face perplexity guide), encoded
with tiktoken's gpt2 encoding. The data file is pinned by dataset revision and
SHA-256. Scoring uses the same sliding-window rule as src/tools/perplexity.cpp:
token t is scored once, from its prefix inside the first window that holds it
past the previous window's end.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import numpy as np  # noqa: E402
import tiktoken  # noqa: E402
import torch  # noqa: E402

DATASET = "Salesforce/wikitext"
DATA_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def slice_ids(revision, tokens):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DATASET, DATA_FILE, repo_type="dataset", revision=revision)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    ids = tiktoken.get_encoding("gpt2").encode_ordinary(text)[:tokens]
    return ids, digest


def score(logits_fn, ids, window, stride):
    nll, scored, windows, previous_end = 0.0, 0, 0, 0
    for begin in range(0, len(ids), stride):
        end = min(len(ids), begin + window)
        logits = logits_fn(ids[begin:end])
        log_probs = torch.log_softmax(logits.double(), dim=-1)
        for t in range(max(begin + 1, previous_end), end):
            nll -= float(log_probs[t - begin - 1, ids[t]])
            scored += 1
        previous_end = end
        windows += 1
        if end == len(ids):
            break
    return dict(
        tokens=len(ids),
        scored=scored,
        windows=windows,
        window=window,
        stride=stride,
        nll_sum=nll,
        perplexity=math.exp(nll / scored),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="dataset revision (commit hash)")
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--ids", type=Path, required=True, help="write the slice's token IDs")
    parser.add_argument("--report", type=Path, help="write the PyTorch result")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    ids, digest = slice_ids(args.revision, args.tokens)
    args.ids.parent.mkdir(parents=True, exist_ok=True)
    args.ids.write_text(" ".join(map(str, ids)), encoding="utf-8")
    ids_digest = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
    print(f"slice: {len(ids)} tokens, parquet sha256 {digest}", flush=True)
    if not args.report:
        return
    from transformers import GPT2LMHeadModel

    manifest = json.loads((ROOT / "golden" / "manifest.json").read_text(encoding="utf-8"))
    torch.set_num_threads(args.threads)
    model = GPT2LMHeadModel.from_pretrained(
        "gpt2", revision=manifest["revision"], attn_implementation="eager", dtype=torch.float32
    ).eval()

    def logits_fn(window_ids):
        with torch.inference_mode():
            return model(torch.tensor([window_ids]), use_cache=False).logits[0]

    result = score(logits_fn, ids, args.window, args.stride)
    result.update(
        runtime="PyTorch float32",
        torch=torch.__version__,
        threads=args.threads,
        dataset=DATASET,
        data_file=DATA_FILE,
        dataset_revision=args.revision,
        parquet_sha256=digest,
        token_ids_sha256=ids_digest,
        model_revision=manifest["revision"],
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
