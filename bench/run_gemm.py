"""Run the GEMM step benchmarks and the FMA peak probe; write results/ JSON and a table.

Peak = cores x clock x FMA units per core x SIMD lanes x 2. The clock is measured
under FMA load from a dependent FMA chain (steps/s x FMA latency in cycles).
FMA units, lanes and latency are stated per microarchitecture and passed in.
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.machine_info import collect  # noqa: E402

SHAPES = [(768, 768, 768), (1024, 3072, 768)]
STEPS = ["naive", "reordered", "blocked", "packed", "avx2"]


def run_benchmark(executable, repetitions, pin_core, extra=()):
    command = [
        str(executable),
        f"--pin-core={pin_core}",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_format=json",
        *extra,
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def iterations(report, prefix):
    rows = {}
    for row in report["benchmarks"]:
        if row.get("run_type") == "iteration" and row["name"].startswith(prefix):
            if row.get("error_occurred"):
                continue
            # Drop the "/real_time" suffix Google Benchmark appends to run names.
            rows.setdefault(row["run_name"].removesuffix("/real_time"), []).append(row)
    return rows


def summary(values):
    return dict(
        median=statistics.median(values),
        minimum=min(values),
        maximum=max(values),
        repetitions=len(values),
    )


def sanitize(value, replacements):
    if isinstance(value, dict) and "host_name" in value:
        value = {**value, "host_name": "<host>"}
    text = json.dumps(value)
    for old, new in replacements:
        for form in (old, old.replace("\\", "\\\\"), old.replace("\\", "/")):
            text = text.replace(form, new)
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build" / "release")
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--pin-core", type=int, default=2)
    parser.add_argument("--fma-units", type=int, default=2, help="FMA pipes per core")
    parser.add_argument("--lanes", type=int, default=8, help="fp32 lanes per FMA (AVX2)")
    parser.add_argument("--fma-latency", type=int, default=4, help="FMA latency, cycles")
    parser.add_argument("--microarchitecture", default="AMD Zen 4")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "m4-gemm.json")
    parser.add_argument("--table", type=Path, default=ROOT / "results" / "m4-gemm.md")
    args = parser.parse_args()
    suffix = ".exe" if sys.platform == "win32" else ""
    build = args.build.resolve()

    peak_report = run_benchmark(build / f"peak_bench{suffix}", args.repetitions, args.pin_core)
    peak_rows = iterations(peak_report, "fma_")
    chain = [row["chain_steps"] for row in peak_rows["fma_latency_chain"]]
    measured_fma = [row["flops"] / 1e9 for row in peak_rows["fma_peak"]]
    clock_ghz = statistics.median(chain) * args.fma_latency / 1e9
    flops_per_cycle = args.fma_units * args.lanes * 2
    peak_gflops = clock_ghz * flops_per_cycle

    gemm_report = run_benchmark(build / f"gemm_bench{suffix}", args.repetitions, args.pin_core)
    rows = iterations(gemm_report, "gemm/")
    measured = {}
    for run_name, repetitions in rows.items():
        step = repetitions[0]["label"]
        _, m, n, k = (part.split(":")[1] for part in run_name.split("/")[1:5])
        gflops = [row["flops"] / 1e9 for row in repetitions]
        milliseconds = [row["real_time"] for row in repetitions]
        measured.setdefault(step, {})[f"{m}x{n}x{k}"] = dict(
            gflops=summary(gflops),
            milliseconds=summary(milliseconds),
            percent_of_single_core_peak=100 * statistics.median(gflops) / peak_gflops,
        )

    replacements = [(str(ROOT), "<project>"), (str(Path.home()), "<user>")]
    report = dict(
        purpose="M4 single-thread GEMM optimization steps",
        machine=sanitize(collect(build), replacements),
        benchmark_context=sanitize(gemm_report["context"], replacements),
        peak=dict(
            formula="cores x clock x FMA units x SIMD lanes x 2",
            cores=1,
            microarchitecture=args.microarchitecture,
            fma_units_per_core=args.fma_units,
            fp32_lanes_per_fma=args.lanes,
            flops_per_fma_lane=2,
            fma_latency_cycles=args.fma_latency,
            clock_ghz_under_fma_load=clock_ghz,
            clock_source="dependent FMA chain: median steps/s x FMA latency",
            theoretical_gflops=peak_gflops,
            measured_fma_throughput_gflops=summary(measured_fma),
        ),
        conditions=dict(
            threads=1,
            pinned_core=args.pin_core,
            flush_to_zero=True,
            turbo="left enabled; changing power settings is out of scope, so the clock is "
            "measured under the same FMA load instead",
        ),
        steps=measured,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    header = [
        f"Single thread, core {args.pin_core}, {args.repetitions} repetitions, median "
        "[min-max] GFLOPS.",
        f"Peak {peak_gflops:.1f} GFLOPS = 1 core x {clock_ghz:.2f} GHz x {args.fma_units} FMA "
        f"x {args.lanes} lanes x 2 (measured FMA probe: "
        f"{statistics.median(measured_fma):.1f} GFLOPS).",
        "",
        "| Step | 768x768x768 | % peak | 1024x3072x768 | % peak | Speedup vs naive |",
        "|---|---|---|---|---|---|",
    ]
    lines = list(header)
    base = measured.get("naive", {}).get("1024x3072x768", {}).get("gflops", {}).get("median")
    for index, step in enumerate(STEPS, 1):
        if step not in measured:
            continue
        cells = []
        for m, n, k in SHAPES:
            row = measured[step][f"{m}x{n}x{k}"]
            g = row["gflops"]
            cells.append(f"{g['median']:.2f} [{g['minimum']:.2f}-{g['maximum']:.2f}]")
            cells.append(f"{row['percent_of_single_core_peak']:.1f}%")
        large = measured[step]["1024x3072x768"]["gflops"]["median"]
        speedup = f"{large / base:.1f}x" if base else "-"
        lines.append(f"| {index}. {step} | " + " | ".join(cells) + f" | {speedup} |")
    args.table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
