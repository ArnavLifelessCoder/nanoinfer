"""Roofline plot from results/: achieved GFLOPS vs arithmetic intensity.

Left: the single-thread GEMM steps against one core's ceilings (m4-gemm.json,
read bandwidth at 1 thread from m5-bandwidth.json). Right: all-core kernels and
end-to-end phases against the machine's ceilings (m7-kernels.json,
m5-bandwidth.json, m7-e2e.json). Arithmetic intensity uses compulsory traffic:
each operand read once and the result written once (4-byte floats; int8 weights
at 1 byte plus 4 bytes per 32-value group). Writes docs/roofline.png and
results/m8-roofline.json with every plotted value.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def gemm_intensity(m, n, k):
    return 2 * m * n * k / (4 * (m * k + k * n + m * n))


def model_costs(manifest_name):
    """Flops and weight bytes per decoded token from an exporter manifest."""
    manifest = json.loads((ROOT / "models" / manifest_name).read_text(encoding="utf-8"))
    flops, weight_bytes = 0, 0
    for tensor in manifest["tensors"]:
        if tensor["name"] == "wpe.weight":
            weight_bytes += tensor["byte_size"] // tensor["shape"][0]  # One row per token.
            continue
        weight_bytes += tensor["byte_size"]
        if len(tensor["shape"]) == 2:  # Matrices, including the tied logits product.
            flops += 2 * tensor["shape"][0] * tensor["shape"][1]
    return flops, weight_bytes


def ceilings(axis, peak, bandwidth, label):
    xs = [0.05, peak / bandwidth, 1000]
    axis.plot(xs, [min(peak, bandwidth * x) for x in xs], "k-", linewidth=1.5)
    axis.text(0.06, bandwidth * 0.07, f"{bandwidth:.1f} GB/s", rotation=33, fontsize=8)
    axis.text(200, peak * 1.08, f"{peak:.0f} GFLOPS {label}", ha="right", fontsize=8)


def main():
    m4, m5, m7k, m7 = (
        load("m4-gemm.json"),
        load("m5-bandwidth.json"),
        load("m7-kernels.json"),
        load("m7-e2e.json"),
    )
    points = {"single_core": [], "all_cores": []}
    for step, shapes in m4["steps"].items():
        for shape, row in shapes.items():
            m, n, k = map(int, shape.split("x"))
            points["single_core"].append(
                dict(
                    label=f"{step} {shape}",
                    ai=gemm_intensity(m, n, k),
                    gflops=row["gflops"]["median"],
                )
            )
    full = max(row["threads"] for row in m7k["gemm_threads"])
    for row in m7k["gemm_threads"]:
        if row["threads"] == full:
            m, n, k = map(int, row["shape"].split("x"))
            points["all_cores"].append(
                dict(
                    label=f"GEMM {row['shape']}",
                    ai=gemm_intensity(m, n, k),
                    gflops=row["gflops"]["median"],
                )
            )
    for row in m5["gemv"]:
        if row["threads"] == full:
            n = int(row["shape"].split("->")[1])
            flops = 2 * 768 * n
            ai = flops / row["weight_bytes"]
            points["all_cores"].append(
                dict(
                    label=f"GEMV {row['shape']} {row['layout']}",
                    ai=ai,
                    gflops=row["gb_per_second"]["median"] * ai,
                )
            )
    for label, manifest in [("runtime f32", "gpt2.json"), ("runtime int8", "gpt2-int8.json")]:
        flops, weight_bytes = model_costs(manifest)
        rate = m7["results"][label][-1]["decode"]["tokens_per_second"]["median"]
        points["all_cores"].append(
            dict(
                label=f"decode {label.split()[1]}",
                ai=flops / weight_bytes,
                gflops=rate * flops / 1e9,
                ceiling_tokens_per_second=m5["measured_bandwidth_gb_per_second"]
                * 1e9
                / weight_bytes,
                measured_tokens_per_second=rate,
                weight_bytes_per_token=weight_bytes,
            )
        )
    single_peak = m4["peak"]["theoretical_gflops"]
    single_bandwidth = m5["read_gb_per_second"]["1"]["median"]
    peak = m7k["peak"]["theoretical_gflops"]
    bandwidth = m5["measured_bandwidth_gb_per_second"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, key, top, bw, title in [
        (axes[0], "single_core", single_peak, single_bandwidth, "One core: GEMM steps"),
        (axes[1], "all_cores", peak, bandwidth, f"{full} cores: kernels and decode"),
    ]:
        ceilings(axis, top, bw, "peak")
        for point in points[key]:
            axis.loglog(point["ai"], point["gflops"], "o")
            axis.annotate(
                point["label"],
                (point["ai"], point["gflops"]),
                fontsize=7,
                xytext=(4, 2),
                textcoords="offset points",
            )
        axis.set_xlabel("Arithmetic intensity (flops per byte)")
        axis.set_ylabel("Achieved GFLOPS")
        axis.set_title(title)
        axis.set_xlim(0.05, 1000)
        axis.set_ylim(0.1, top * 3)
        axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    figure.savefig(ROOT / "docs" / "roofline.png", dpi=150)
    report = dict(
        purpose="values plotted in docs/roofline.png",
        single_core=dict(peak_gflops=single_peak, read_bandwidth_gb_per_second=single_bandwidth),
        all_cores=dict(peak_gflops=peak, read_bandwidth_gb_per_second=bandwidth, threads=full),
        points=points,
    )
    (ROOT / "results/m8-roofline.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["points"]["all_cores"], indent=2))


if __name__ == "__main__":
    main()
