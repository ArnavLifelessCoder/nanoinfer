"""M7 end-to-end benchmarks: runtime f32 and int8 vs PyTorch CPU at equal thread counts.

Measures prefill (128 and 512 tokens), decode (128 tokens after a 32-token
prompt, tokens/s and ms/token p50/p99) and thread scaling, with the same token
IDs for every system. Writes results/m7-e2e.json and results/m7.md. Refuses to
run on battery power unless --allow-battery is given, and records the power
state and background CPU load, because both move these numbers a lot.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.common import ROOT, sanitize, write_ids  # noqa: E402
from bench.run_m5 import e2e, scaling_row  # noqa: E402
from scripts.machine_info import collect  # noqa: E402


def power_state():
    if sys.platform != "win32":
        return dict(note="not recorded on this platform")
    script = (
        "$b = Get-CimInstance Win32_Battery | Select-Object -First 1;"
        "$load = (Get-Counter '\\Processor(_Total)\\% Processor Time' -SampleInterval 2 "
        "-MaxSamples 1).CounterSamples.CookedValue;"
        "[ordered]@{on_battery = ($b -ne $null -and $b.BatteryStatus -ne 2);"
        "battery_percent = $b.EstimatedChargeRemaining; background_cpu_percent = $load;"
        "power_scheme = (powercfg /getactivescheme)} | ConvertTo-Json"
    )
    output = subprocess.run(
        ["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True
    ).stdout
    return json.loads(output)


def pytorch(ids, threads, repetitions, prefill, prompt, new):
    command = [
        sys.executable,
        str(ROOT / "bench" / "pytorch_baseline.py"),
        "--ids",
        str(ids),
        "--threads",
        str(threads),
        "--repetitions",
        str(repetitions),
        "--prefill",
        *map(str, prefill),
        "--decode-prompt",
        str(prompt),
        "--decode-tokens",
        str(new),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build" / "release")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--allow-battery", action="store_true")
    args = parser.parse_args()
    before = power_state()
    if before.get("on_battery") and not args.allow_battery:
        raise SystemExit("on battery power: plug in, or pass --allow-battery and say so")
    build = args.build.resolve()
    ids = write_ids(build / "bench-prompt-ids.txt", 512)
    models = {
        "runtime f32": ROOT / "models/gpt2.bin",
        "runtime int8": ROOT / "models/gpt2-int8.bin",
    }
    full = max(args.threads)
    results = {"PyTorch f32": [], "runtime f32": [], "runtime int8": []}
    for threads in args.threads:
        prefill = [128, 512] if threads == full else [128]
        for label, model in models.items():
            raw = e2e(build, model, ids, threads, "avx2", args.repetitions, prefill, 32, 128)
            results[label].append(scaling_row(raw))
        raw = pytorch(ids, threads, args.repetitions, prefill, 32, 128)
        results["PyTorch f32"].append(scaling_row(raw) | dict(torch=raw["torch"]))
        print(f"done {threads} threads", flush=True)
    after = power_state()
    report = dict(
        purpose="M7 end-to-end: runtime f32 and int8 vs PyTorch CPU, same prompts and threads",
        machine=sanitize(collect(build)),
        power_before=before,
        power_after=after,
        prompt="bench.common.PROMPT_TEXT tokens; identical IDs for every system",
        phases=dict(
            prefill="whole prompt in one call, cache filled, logits for every position",
            decode="128 greedy steps after a 32-token prompt; each step timed alone",
        ),
        pytorch_settings="float32, torch.inference_mode(), eager attention, "
        "torch.set_num_threads(threads), HF KV cache",
        runtime_settings="avx2 kernels, one pinned thread per physical core up to threads",
        repetitions=args.repetitions,
        results=results,
    )
    (ROOT / "results/m7-e2e.json").write_text(json.dumps(report, indent=2) + "\n")

    def cell(row, phase, tokens=None):
        if phase == "decode":
            return row["decode"]["tokens_per_second"]["median"]
        return row["prefill"][str(tokens)]["tokens_per_second"]["median"]

    lines = [f"All rows at {full} threads unless noted; medians of {args.repetitions} runs.", ""]
    lines += [
        "| System | Prefill 128 tok/s | Prefill 512 tok/s | Decode tok/s | ms/token p50 | p99 |",
        "|---|---|---|---|---|---|",
    ]
    for label, rows in results.items():
        row = rows[-1]
        ms = row["decode"]["ms_per_token"]
        lines.append(
            f"| {label} | {cell(row, 'prefill', 128):.1f} | {cell(row, 'prefill', 512):.1f} | "
            f"{cell(row, 'decode'):.1f} | {ms['median']:.2f} | {ms['p99']:.2f} |"
        )
    lines += ["", "Thread scaling (prefill 128 / decode tok/s):", ""]
    lines += ["| Threads | " + " | ".join(results) + " |", "|---" * (len(results) + 1) + "|"]
    for index, threads in enumerate(args.threads):
        cells = [
            f"{cell(rows[index], 'prefill', 128):.1f} / {cell(rows[index], 'decode'):.1f}"
            for rows in results.values()
        ]
        lines.append(f"| {threads} | " + " | ".join(cells) + " |")
    lines += [
        "",
        f"Power before: on battery {before.get('on_battery')}, background CPU "
        f"{before.get('background_cpu_percent', 0):.0f}%; after: on battery "
        f"{after.get('on_battery')}.",
    ]
    (ROOT / "results/m7.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
