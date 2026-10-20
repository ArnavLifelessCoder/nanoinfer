"""Check reference execution boundaries, causal prefixes and absolute positions."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from python.export import export_weights, make_tiny


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = make_tiny(args.seed)
    directory = args.directory.resolve() / "contract \u03bb"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.bin"
    export_weights(model, path)
    executable = str(args.executable.resolve())
    print(f"Random seed: {args.seed}", flush=True)

    def run(label, tokens, new_tokens=0, success=True):
        ids = directory / f"{label}.txt"
        output = directory / label
        ids.write_text(" ".join(map(str, tokens)), encoding="utf-8")
        result = subprocess.run(
            [executable, str(path), str(ids), str(output), str(new_tokens)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == (0 if success else 1), (label, result.stderr, result.stdout)
        if not success:
            assert "reference error:" in result.stderr and not result.stdout
            print(f"PASS reject {label}", flush=True)
            return None
        metadata = json.loads((output / "output.json").read_text(encoding="utf-8"))
        values = np.fromfile(output / "logits.f32", dtype="<f4").reshape(
            len(tokens), model.config.vocab_size
        )
        return values, metadata, output

    generator = np.random.default_rng(args.seed)
    tokens = generator.integers(0, model.config.vocab_size, model.config.n_positions).tolist()
    full, metadata, output = run("full-context", tokens)
    assert metadata["greedy_ids"] == []
    with torch.inference_mode():
        expected = model(torch.tensor([tokens]), use_cache=False, output_hidden_states=True)
    np.testing.assert_allclose(full, expected.logits[0].numpy(), atol=1e-5, rtol=0)
    final_ln = np.fromfile(output / "final_ln.f32", dtype="<f4").reshape(len(tokens), -1)
    np.testing.assert_allclose(final_ln, expected.hidden_states[-1][0].numpy(), atol=1e-5, rtol=0)
    for length in (1, 7, 63):
        prefix, _, _ = run(f"prefix-{length}", tokens[:length])
        np.testing.assert_array_equal(prefix, full[:length])
    print("PASS full context, absolute positions and causal prefix invariance", flush=True)

    _, generated, _ = run("context-edge", tokens[:-1], new_tokens=1)
    assert generated["greedy_ids"] == [int(full[-2].argmax())]
    for label, ids, count in (
        ("empty", [], 0),
        ("negative", [-1], 0),
        ("out-of-vocabulary", [model.config.vocab_size], 0),
        ("integer-overflow", [4294967296], 0),
        ("non-integer", ["1x"], 0),
        ("too-long", tokens + [1], 0),
        ("continuation-overflow", tokens, 1),
        ("negative-count", [1], -1),
        ("count-overflow", [1], 4294967296),
    ):
        run(label, ids, count, success=False)
    missing_args = subprocess.run([executable], capture_output=True, text=True)
    assert missing_args.returncode == 2 and "usage:" in missing_args.stderr

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    export_weights(model, path)
    _, generated, _ = run("argmax-ties", [1, 2], new_tokens=3)
    assert generated["greedy_ids"] == [0, 0, 0]
    print("PASS greedy chooses lowest ID on ties, emits requested count and has no EOS stop")


if __name__ == "__main__":
    main()
