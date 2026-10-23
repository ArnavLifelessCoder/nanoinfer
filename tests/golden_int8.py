"""Check the int8 kernels against the int8 reference twin on the golden prompts.

The reference twin is the `naive` step, which multiplies the exactly
dequantized float32 weights. The packed and avx2 int8 kernels dequantize in
their inner loops and must agree with it within the approved bounds (logits
1e-3; hidden atol 1e-4 + rtol 1e-6). Greedy agreement with the float32 PyTorch
golden is reported, not required: int8 may change tokens.

M4's float64 rule, applied to int8: when a hidden tensor is outside the bound,
it still passes if its error against the float64 oracle of the int8 weights
(golden/oracle_int8_prompt_N.npz, python/oracle.py --quant int8) is within the
float32 rounding noise of a correct implementation: no larger than the larger
of the reference twin's error against that oracle and PyTorch float32's error
against its own float64 oracle (golden/oracle_prompt_N.npz), plus 1e-4. A
dequantization or indexing bug produces errors far above that noise.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from tests.golden_forward import compare_array
from tests.native_process import capture


def run(executable, model, directory, prompt, kernel, threads, new_tokens, timeout):
    output = directory / f"{kernel}-t{threads}" / Path(prompt["file"]).stem
    output.mkdir(parents=True, exist_ok=True)
    ids = output / "ids.txt"
    ids.write_text(" ".join(map(str, prompt["ids"])), encoding="utf-8")
    result = capture(
        [str(executable), str(model), str(ids), str(output), str(new_tokens), kernel, str(threads)],
        timeout=timeout,
    )
    assert result.returncode == 0, (kernel, result.stderr)
    return output


def load(output, key):
    return np.fromfile(output / f"{key}.f32", dtype="<f4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--kernels", nargs="+", default=["packed", "avx2"])
    parser.add_argument("--reference-kernel", default="naive")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    executable, model = args.executable.resolve(), args.model.resolve()
    directory = args.directory.resolve()
    golden_dir = args.golden.resolve()
    manifest = json.loads((golden_dir / "manifest.json").read_text(encoding="utf-8"))
    rows, failures, agreements = [], [], []
    for prompt in manifest["prompts"]:
        common = (executable, model, directory, prompt)
        reference = run(*common, args.reference_kernel, 1, args.new_tokens, args.timeout)
        reference_meta = json.loads((reference / "output.json").read_text(encoding="utf-8"))
        keys = ["logits", "final_ln"] + [f"layer_{i}" for i in range(reference_meta["layers"])]
        with np.load(golden_dir / prompt["file"], allow_pickle=False) as golden:
            torch_tokens = golden["greedy_ids"][: args.new_tokens].tolist()
        oracle_path = golden_dir / f"oracle_int8_{prompt['file']}"
        oracle, torch_noise = {}, {}
        if oracle_path.exists():
            with np.load(oracle_path, allow_pickle=False) as arrays:
                oracle = {key: arrays[key] for key in arrays.files}
            with (
                np.load(golden_dir / prompt["file"], allow_pickle=False) as golden,
                np.load(golden_dir / f"oracle_{prompt['file']}", allow_pickle=False) as f64,
            ):
                torch_noise = {
                    key: float(np.max(np.abs(golden[key].astype(np.float64) - f64[key])))
                    for key in f64.files
                }
        for kernel in args.kernels:
            output = run(*common, kernel, 1, args.new_tokens, args.timeout)
            threaded = run(*common, kernel, args.threads, args.new_tokens, args.timeout)
            meta = json.loads((output / "output.json").read_text(encoding="utf-8"))
            row = dict(file=prompt["file"], kernel=kernel, tensors={})
            for key in keys:
                actual, target = load(output, key), load(reference, key)
                difference, ratio, passed = compare_array(actual, target, hidden=key != "logits")
                row["tensors"][key] = dict(max_abs_diff=difference, tolerance_ratio=ratio)
                if not passed and key in oracle:
                    truth = oracle[key].reshape(-1)
                    kernel_error = float(np.max(np.abs(actual - truth)))
                    reference_error = float(np.max(np.abs(target - truth)))
                    noise = max(reference_error, torch_noise[key])
                    row["tensors"][key].update(
                        error_vs_float64=kernel_error,
                        reference_error_vs_float64=reference_error,
                        torch_float32_error_vs_float64=torch_noise[key],
                    )
                    passed = kernel_error <= noise + 1e-4
                if not passed:
                    failures.append(f"{kernel} {prompt['file']} {key} outside the bound")
                if load(threaded, key).tobytes() != actual.tobytes():
                    failures.append(f"{kernel} {prompt['file']} {key} depends on threads")
            if meta["greedy_ids"] != reference_meta["greedy_ids"]:
                failures.append(f"{kernel} {prompt['file']} greedy differs from the reference")
            agree = sum(a == b for a, b in zip(meta["greedy_ids"], torch_tokens, strict=True))
            row["greedy_agreement_with_float32_torch"] = agree / len(torch_tokens)
            agreements.append(agree / len(torch_tokens))
            rows.append(row)
            print(
                f"{kernel} {prompt['file']}: greedy agreement with float32 torch {agree}/"
                f"{len(torch_tokens)}",
                flush=True,
            )
    report = dict(
        rule=__doc__.strip(),
        model=str(args.model.name),
        reference_kernel=args.reference_kernel,
        rows=rows,
        mean_greedy_agreement=sum(agreements) / len(agreements),
        failures=failures,
        passed=not failures,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    assert not failures, failures
    print("PASS int8 kernels match the int8 reference twin at every thread count", flush=True)


if __name__ == "__main__":
    main()
