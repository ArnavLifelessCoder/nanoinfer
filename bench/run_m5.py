"""M5 benchmarks: STREAM triad bandwidth, decode GEMV bandwidth and thread scaling.

Writes results/m5-bandwidth.json, results/m5-scaling.json and results/m5.md.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.common import (
    ROOT,
    SUFFIX,
    power_state,
    run_google_benchmark,
    sanitize,
    summary,
    write_ids,
)  # noqa: E402
from scripts.machine_info import collect  # noqa: E402


def e2e(build, model, ids, threads, gemm, repetitions, prefill, prompt, new):
    command = [
        str(build / f"e2e_bench{SUFFIX}"),
        str(model),
        str(ids),
        str(threads),
        gemm,
        str(repetitions),
        ",".join(map(str, prefill)),
        str(prompt),
        str(new),
    ]
    return json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)


def scaling_row(raw):
    row = dict(threads=raw["threads"], prefill={}, decode={})
    for entry in raw["prefill"]:
        rates = [entry["tokens"] / s for s in entry["seconds"]]
        row["prefill"][str(entry["tokens"])] = dict(tokens_per_second=summary(rates))
    runs = raw["decode"]["runs"]
    rates = [len(run) / sum(run) for run in runs]
    steps = [1000 * s for run in runs for s in run]
    row["decode"] = dict(tokens_per_second=summary(rates), ms_per_token=summary(steps))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build" / "release")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "gpt2.bin")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    build = args.build.resolve()
    power = power_state()
    ids = write_ids(build / "bench-prompt-ids.txt", 512)

    _, stream_rows = run_google_benchmark(build / f"stream_bench{SUFFIX}", 5)
    triad, read = {}, {}
    for name, rows in stream_rows.items():
        target = triad if name.startswith("triad") else read
        target[int(rows[0]["threads"])] = summary([row["bytes"] / 1e9 for row in rows])
    # Decode only reads weights, so the read-only stream is its bandwidth ceiling.
    measured_bandwidth = max(entry["median"] for entry in read.values())

    gemv_report, gemv_rows = run_google_benchmark(build / f"gemv_bench{SUFFIX}", 7)
    gemv = []
    for name, rows in gemv_rows.items():
        parts = dict(part.split(":") for part in name.split("/")[1:3])
        rates = summary([row["bytes"] / 1e9 for row in rows])
        gemv.append(
            dict(
                shape=f"768->{parts['n']}",
                layout=rows[0]["label"],
                threads=int(parts["threads"]),
                weight_bytes=int(
                    768 * int(parts["n"]) * (4 if "f32" in rows[0]["label"] else 1.125)
                ),
                gb_per_second=rates,
                percent_of_measured_bandwidth=100 * rates["median"] / measured_bandwidth,
            )
        )

    scaling = [
        scaling_row(
            e2e(build, args.model.resolve(), ids, t, "avx2", args.repetitions, [128], 32, 128)
        )
        for t in args.threads
    ]

    machine = sanitize(collect(build))
    bandwidth = dict(
        purpose="M5 memory bandwidth (STREAM triad) and decode GEMV bandwidth",
        machine=machine,
        power=power,
        context=sanitize(gemv_report["context"]),
        triad_bytes_counted="3 x 4 bytes per element (STREAM convention; write-allocate not counted)",
        triad_gb_per_second={str(k): v for k, v in sorted(triad.items())},
        read_bytes_counted="4 bytes per element read once (AVX2 sum, eight accumulators)",
        read_gb_per_second={str(k): v for k, v in sorted(read.items())},
        measured_bandwidth_definition="best median of the read-only stream over thread counts",
        measured_bandwidth_gb_per_second=measured_bandwidth,
        gemv=gemv,
        note="768->3072 weights are 9.4 MB and fit in L3, so that row measures cache "
        "bandwidth; 768->50257 (154 MB) measures DRAM.",
    )
    (ROOT / "results/m5-bandwidth.json").write_text(json.dumps(bandwidth, indent=2) + "\n")
    scaling_report = dict(
        purpose="M5 thread scaling: prefill 128 tokens; decode 128 tokens after a 32-token prompt",
        machine=machine,
        power=power,
        kernel="avx2",
        repetitions=args.repetitions,
        prompt="bench.common.PROMPT_TEXT tokens, identical for every run",
        pinning="participant i pinned to physical core i; one thread per physical core",
        rows=scaling,
    )
    (ROOT / "results/m5-scaling.json").write_text(json.dumps(scaling_report, indent=2) + "\n")

    lines = [
        f"Measured bandwidth (read-only stream, best thread count): {measured_bandwidth:.1f} GB/s.",
        "",
        "| Threads | Read GB/s (median) | Triad GB/s (median) |",
        "|---|---|---|",
    ]
    lines += [f"| {k} | {read[k]['median']:.1f} | {triad[k]['median']:.1f} |" for k in sorted(read)]
    lines += [
        "",
        "| GEMV | Threads | GB/s (median) | % of measured bandwidth |",
        "|---|---|---|---|",
    ]
    for row in sorted(gemv, key=lambda r: (r["shape"], r["threads"])):
        lines.append(
            f"| {row['shape']} {row['layout']} | {row['threads']} | "
            f"{row['gb_per_second']['median']:.1f} | {row['percent_of_measured_bandwidth']:.0f}% |"
        )
    lines += [
        "",
        "| Threads | Prefill 128 tok/s | Decode tok/s | Decode ms/token p50 | p99 |",
        "|---|---|---|---|---|",
    ]
    for row in scaling:
        p = row["prefill"]["128"]["tokens_per_second"]["median"]
        d = row["decode"]
        lines.append(
            f"| {row['threads']} | {p:.1f} | {d['tokens_per_second']['median']:.1f} | "
            f"{d['ms_per_token']['median']:.2f} | {d['ms_per_token']['p99']:.2f} |"
        )
    (ROOT / "results/m5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
