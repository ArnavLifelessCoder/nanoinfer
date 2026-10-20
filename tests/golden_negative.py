"""Require golden rejection of test-only GELU and LayerNorm mutations."""

import argparse
import json
from pathlib import Path

import torch

from python.export import export_weights, make_tiny, write_golden
from tests.golden_forward import check_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--erf-executable", type=Path, required=True)
    parser.add_argument("--epsilon-executable", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    directory = args.directory.resolve()
    torch.set_num_threads(1)
    model = make_tiny(42)
    # Unscaled random tiny weights keep GELU near zero, where erf and tanh
    # nearly coincide. Exercise the curved part of GELU with a fixed fixture.
    with torch.no_grad():
        for block in model.transformer.h:
            block.mlp.c_fc.weight.mul_(20)
            block.mlp.c_proj.weight.mul_(5)
    model_path = directory / "tiny-sensitive.bin"
    golden = directory / "golden"
    export_weights(model, model_path)
    write_golden(model, golden, tiny=True)
    manifest = json.loads((golden / "manifest.json").read_text(encoding="utf-8"))
    results = {}
    for name, executable in (
        ("default", args.executable),
        ("exact_erf_gelu", args.erf_executable),
        ("layer_norm_eps_1e_6", args.epsilon_executable),
    ):
        results[name] = [
            check_prompt(
                executable.resolve(), model_path, golden, directory / name, prompt, new_tokens=0
            )
            for prompt in manifest["prompts"]
        ]
    default_passed = all(not prompt["failures"] for prompt in results["default"])
    rejected = {
        name: all(bool(prompt["failures"]) for prompt in prompts)
        for name, prompts in results.items()
        if name != "default"
    }
    report = dict(
        fixture="seed 42 tiny GPT-2; MLP fc weights x20, projection weights x5",
        reason="exercise nonlinear GELU curvature rather than only its near-zero regime",
        reference_torch=torch.__version__,
        tolerance=dict(logits_max_abs_exclusive=1e-3, hidden_atol=1e-4, hidden_rtol=1e-6),
        default_passed=default_passed,
        rejected_on_every_prompt=rejected,
        runs=results,
        passed=default_passed and all(rejected.values()),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert default_passed, "correct default forward must pass the same golden comparison"
    for name, did_reject in rejected.items():
        assert did_reject, f"{name} was not rejected on every prompt"
    print("PASS default forward; golden comparison rejects exact erf GELU and LayerNorm eps 1e-6")


if __name__ == "__main__":
    main()
