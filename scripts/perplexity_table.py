"""Runtime f32 and int8 perplexity on the slice python/perplexity.py wrote; M6 table.

Requires results/m6-perplexity-pytorch.json and the token IDs file from
`python -m python.perplexity --ids ... --report results/m6-perplexity-pytorch.json`.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = ".exe" if sys.platform == "win32" else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build" / "release")
    parser.add_argument("--ids", type=Path, default=ROOT / "build" / "wikitext2-slice-ids.txt")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    pytorch = json.loads((ROOT / "results/m6-perplexity-pytorch.json").read_text(encoding="utf-8"))
    ids = [int(token) for token in args.ids.read_text(encoding="utf-8").split()]
    digest = hashlib.sha256(np.asarray(ids, dtype="<u4").tobytes()).hexdigest()
    assert digest == pytorch["token_ids_sha256"], "token IDs differ from the PyTorch run"
    rows = [dict(config="PyTorch f32", perplexity=pytorch["perplexity"], scored=pytorch["scored"])]
    runtime = {}
    for label, model in [("runtime f32", "gpt2.bin"), ("runtime int8", "gpt2-int8.bin")]:
        command = [
            str(args.build / f"gpt2ppl{SUFFIX}"),
            str(ROOT / "models" / model),
            str(args.ids),
            str(pytorch["window"]),
            str(pytorch["stride"]),
            str(args.threads),
            "avx2",
        ]
        result = json.loads(
            subprocess.run(command, capture_output=True, text=True, check=True).stdout
        )
        assert result["scored"] == pytorch["scored"] and result["windows"] == pytorch["windows"]
        runtime[label] = result
        rows.append(dict(config=label, perplexity=result["perplexity"], scored=result["scored"]))
    base = rows[0]["perplexity"]
    report = dict(
        purpose="M6 perplexity: PyTorch f32, runtime f32, runtime int8 on the same slice",
        slice=dict(
            dataset=pytorch["dataset"],
            data_file=pytorch["data_file"],
            dataset_revision=pytorch["dataset_revision"],
            parquet_sha256=pytorch["parquet_sha256"],
            tokens=pytorch["tokens"],
            token_ids_sha256=digest,
            window=pytorch["window"],
            stride=pytorch["stride"],
            scoring="token t scored once from its in-window prefix (src/tools/perplexity.cpp)",
        ),
        int8="symmetric, group 32 along the input dimension; wte, c_attn, attn.c_proj, c_fc, "
        "mlp.c_proj quantized; wpe, LayerNorm and biases f32",
        rows=rows,
        runtime_details=runtime,
        int8_minus_runtime_f32=runtime["runtime int8"]["perplexity"]
        - runtime["runtime f32"]["perplexity"],
    )
    (ROOT / "results/m6-perplexity.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"WikiText-2 raw test, first {pytorch['tokens']} tokens, window {pytorch['window']}, "
        f"stride {pytorch['stride']}, {pytorch['scored']} tokens scored.",
        "",
        "| Configuration | Perplexity | vs PyTorch f32 |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['config']} | {row['perplexity']:.4f} | {row['perplexity'] - base:+.4f} |"
        )
    (ROOT / "results/m6-perplexity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
