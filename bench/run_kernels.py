"""All-core FMA peak and threaded GEMM (step 6) scaling; results/m7-kernels.json and .md.

Peak = cores x clock x FMA units x lanes x 2, with the clock measured on every
physical core at once from a dependent FMA chain (steps/s x FMA latency).
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.common import ROOT, SUFFIX, run_google_benchmark, sanitize, summary  # noqa: E402
from scripts.machine_info import collect  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build" / "release")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--fma-units", type=int, default=2)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--fma-latency", type=int, default=4)
    args = parser.parse_args()
    build = args.build.resolve()
    _, peak = run_google_benchmark(build / f"peak_bench{SUFFIX}", args.repetitions)
    cores = int(peak["fma_all_cores_chain"][0]["threads"])
    clock = statistics.median(r["chain_steps"] for r in peak["fma_all_cores_chain"]) * (
        args.fma_latency / 1e9
    )
    theoretical = cores * clock * args.fma_units * args.lanes * 2
    measured = summary([r["flops"] / 1e9 for r in peak["fma_all_cores"]])

    report_raw, rows = run_google_benchmark(
        build / f"gemm_bench{SUFFIX}", args.repetitions, extra=["--benchmark_filter=gemm_threads"]
    )
    gemm = []
    for name, repetitions in rows.items():
        parts = dict(part.split(":") for part in name.split("/")[1:5])
        g = summary([r["flops"] / 1e9 for r in repetitions])
        threads = int(parts["threads"])
        gemm.append(
            dict(
                shape=f"{parts['m']}x{parts['n']}x{parts['k']}",
                threads=threads,
                gflops=g,
                percent_of_peak_for_threads=100 * g["median"] / (threads * clock * 32),
                percent_of_all_core_peak=100 * g["median"] / theoretical,
            )
        )
    gemm.sort(key=lambda row: (row["shape"], row["threads"]))
    report = dict(
        purpose="M7 all-core compute ceiling and threaded GEMM (step 6)",
        machine=sanitize(collect(build)),
        context=sanitize(report_raw["context"]),
        peak=dict(
            formula="cores x clock x FMA units x SIMD lanes x 2",
            cores=cores,
            clock_ghz_all_cores_under_fma_load=clock,
            fma_units_per_core=args.fma_units,
            fp32_lanes_per_fma=args.lanes,
            fma_latency_cycles=args.fma_latency,
            theoretical_gflops=theoretical,
            measured_all_core_fma_gflops=measured,
        ),
        gemm_threads=gemm,
    )
    (ROOT / "results/m7-kernels.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        f"All-core peak {theoretical:.0f} GFLOPS = {cores} cores x {clock:.2f} GHz x "
        f"{args.fma_units} FMA x {args.lanes} lanes x 2 (measured FMA probe "
        f"{measured['median']:.0f} GFLOPS).",
        "",
        "| GEMM | Threads | GFLOPS (median) | % of peak for those threads | % of all-core peak |",
        "|---|---|---|---|---|",
    ]
    for row in gemm:
        lines.append(
            f"| {row['shape']} | {row['threads']} | {row['gflops']['median']:.1f} | "
            f"{row['percent_of_peak_for_threads']:.0f}% | {row['percent_of_all_core_peak']:.0f}% |"
        )
    (ROOT / "results/m7-kernels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
