"""Write per-prompt, per-layer diagnostics from native captures and PyTorch goldens."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def hidden_error_stats(actual, target):
    assert actual.shape == target.shape and target.ndim == 2 and target.size
    assert np.isfinite(actual).all() and np.isfinite(target).all()
    target_abs = np.abs(target.astype(np.float64))
    error = np.abs(actual.astype(np.float64) - target.astype(np.float64))
    below = target_abs < 1024
    relative = target_abs >= 1
    ratio = error / (1e-4 + 1e-6 * target_abs)

    def coordinate(flat_index):
        position, channel = np.unravel_index(flat_index, target.shape)
        return dict(
            position=int(position),
            channel=int(channel),
            torch_abs=float(target_abs[position, channel]),
        )

    return dict(
        max_abs_diff=float(error.max()),
        max_abs_diff_torch_abs_lt_1024=float(error[below].max()) if below.any() else None,
        max_relative_diff_torch_abs_ge_1=(
            float((error[relative] / target_abs[relative]).max()) if relative.any() else None
        ),
        count_abs_diff_gt_1e_4=int(np.count_nonzero(error > 1e-4)),
        worst_element=coordinate(error.argmax()),
        max_tolerance_ratio=float(ratio.max()),
        worst_tolerance_element=coordinate(ratio.argmax()),
        elements=int(target.size),
        elements_torch_abs_lt_1024=int(below.sum()),
        elements_torch_abs_ge_1=int(relative.sum()),
    )


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.golden / "manifest.json").read_text(encoding="utf-8"))
    prompts = []
    for prompt in manifest["prompts"]:
        source = args.golden / prompt["file"]
        directory = args.captures / source.stem
        metadata = json.loads((directory / "output.json").read_text(encoding="utf-8"))
        assert metadata["positions"] == len(prompt["ids"])
        layers = {}
        hashes = {}
        with np.load(source, allow_pickle=False) as golden:
            keys = [f"layer_{index}" for index in range(metadata["layers"])] + ["final_ln"]
            assert set(keys) == {key for key in golden.files if key.startswith("layer_")} | {
                "final_ln"
            }
            for key in keys:
                capture = directory / f"{key}.f32"
                target = golden[key]
                actual = np.fromfile(capture, dtype="<f4").reshape(target.shape)
                layers[key] = hidden_error_stats(actual, target)
                hashes[key] = digest(capture)
        prompts.append(
            dict(
                file=prompt["file"],
                golden_sha256=digest(source),
                capture_sha256=hashes,
                layers=layers,
            )
        )
    report = dict(
        model=manifest["model"],
        model_revision=manifest["revision"],
        reference_torch=manifest["torch"],
        semantics={
            "indices": "zero-based prompt position and hidden channel",
            "worst_element": "largest absolute error; first row-major element breaks ties",
            "relative_error": "abs(actual - torch) / abs(torch), restricted to abs(torch) >= 1",
            "empty_subset": "null maximum with zero subset count",
            "below_1024": "diagnostic only; strict absolute 1e-4 condition withdrawn by owner",
        },
        prompts=prompts,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PASS statistics for {len(prompts)} prompts, all block outputs and final_ln")


if __name__ == "__main__":
    main()
