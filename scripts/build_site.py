"""Build the static project site from results/ (standard library only).

    python scripts/build_site.py --output site

Every number on the page is read from the committed JSON in results/, the same
files the README cites. The page is self-contained HTML with inline SVG charts;
the roofline plot and demo screenshot are copied next to it.
"""

import argparse
import html
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/ArnavLifelessCoder/nanoinfer"


def load(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def bar_chart(rows, unit, width=640, bar=26, gap=12, label_width=190, fmt="{:.1f}"):
    """Horizontal bars as inline SVG; rows are (label, value, css class)."""
    top = max(value for _, value, _ in rows) * 1.18
    height = len(rows) * (bar + gap) + gap
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" class="chart" '
        f'aria-label="{html.escape(unit)}">'
    ]
    for index, (label, value, kind) in enumerate(rows):
        y = gap + index * (bar + gap)
        length = (width - label_width - 70) * value / top
        parts.append(
            f'<text x="{label_width - 10}" y="{y + bar * 0.68}" text-anchor="end" '
            f'class="label">{html.escape(label)}</text>'
            f'<rect x="{label_width}" y="{y}" width="{length:.1f}" height="{bar}" rx="4" '
            f'class="{kind}"/>'
            f'<text x="{label_width + length + 8:.1f}" y="{y + bar * 0.68}" class="value">'
            f"{fmt.format(value)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "site")
    args = parser.parse_args()
    m4, m5 = load("m4-gemm.json"), load("m5-bandwidth.json")
    kernels, e2e = load("m7-kernels.json"), load("m7-e2e.json")
    perplexity, roofline = load("m6-perplexity.json"), load("m8-roofline.json")

    steps = m4["steps"]
    gemm_rows = [
        (f"{index}. {step}", steps[step]["1024x3072x768"]["gflops"]["median"], "bar")
        for index, step in enumerate(steps, 1)
    ]
    threaded = max(
        (r for r in kernels["gemm_threads"] if r["shape"] == "1024x3072x768"),
        key=lambda r: r["threads"],
    )
    gemm_rows.append(
        (f"6. avx2, {threaded['threads']} threads", threaded["gflops"]["median"], "accent")
    )

    def best(label):
        return max(e2e["results"][label], key=lambda r: r["decode"]["tokens_per_second"]["median"])

    def at(label, threads):
        return next(r for r in e2e["results"][label] if r["threads"] == threads)

    int8, f32 = best("runtime int8"), best("runtime f32")
    torch_same = at("PyTorch f32", int8["threads"])
    decode_rows = [
        (
            f"runtime int8 ({int8['threads']} thr)",
            int8["decode"]["tokens_per_second"]["median"],
            "accent",
        ),
        (
            f"runtime f32 ({f32['threads']} thr)",
            f32["decode"]["tokens_per_second"]["median"],
            "bar",
        ),
    ] + [
        (
            f"PyTorch f32 ({row['threads']} thr)",
            row["decode"]["tokens_per_second"]["median"],
            "muted",
        )
        for row in e2e["results"]["PyTorch f32"]
        if row["threads"] in (int8["threads"], f32["threads"])
    ]
    speedup = (
        int8["decode"]["tokens_per_second"]["median"]
        / torch_same["decode"]["tokens_per_second"]["median"]
    )
    decode_points = {
        p["label"].split(" (")[0]: p
        for p in roofline["points"]["all_cores"]
        if p["label"].startswith("decode")
    }
    ppl = {row["config"]: row["perplexity"] for row in perplexity["rows"]}
    avx2 = steps["avx2"]["1024x3072x768"]
    peak = m4["peak"]

    cards = [
        (
            f"{int8['decode']['tokens_per_second']['median']:.1f} tok/s",
            f"int8 decode, {speedup:.1f}x PyTorch at {int8['threads']} threads",
        ),
        (
            f"{avx2['percent_of_single_core_peak']:.0f}% of peak",
            f"AVX2 GEMM on one core, {avx2['gflops']['median']:.1f} GFLOPS",
        ),
        (
            f"{threaded['gflops']['median']:.0f} GFLOPS",
            f"GEMM on {threaded['threads']} cores, {threaded['percent_of_all_core_peak']:.0f}% of peak",
        ),
        (
            f"+{ppl['runtime int8'] - ppl['runtime f32']:.2f} perplexity",
            f"int8 vs f32 on WikiText-2 ({ppl['runtime int8']:.2f} vs {ppl['runtime f32']:.2f})",
        ),
    ]
    card_html = "".join(
        f'<div class="card"><div class="big">{html.escape(big)}</div>'
        f'<div class="small">{html.escape(small)}</div></div>'
        for big, small in cards
    )
    ppl_rows = "".join(
        f"<tr><td>{html.escape(row['config'])}</td><td>{row['perplexity']:.4f}</td></tr>"
        for row in perplexity["rows"]
    )
    ceiling_rows = "".join(
        f"<tr><td>{name}</td><td>{p['measured_tokens_per_second']:.1f}</td>"
        f"<td>{p['ceiling_tokens_per_second']:.1f}</td>"
        f"<td>{100 * p['measured_tokens_per_second'] / p['ceiling_tokens_per_second']:.0f}%</td></tr>"
        for name, p in decode_points.items()
    )
    machine = e2e["machine"]["system"]
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>nanoinfer</title>
<meta name="description" content="A GPT-2 inference runtime for CPUs with hand-written AVX2 kernels, int8 weights and a roofline analysis.">
<style>
:root {{ --bg:#f6f7f9; --panel:#fff; --text:#1c2129; --muted:#667085; --border:#e3e6eb;
  --accent:#2f6fde; --bar:#7aa2e8; --dim:#c3cad6; --mono:ui-monospace,"Cascadia Mono",Consolas,monospace; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0f1115; --panel:#171a20; --text:#e6e8ec;
  --muted:#9aa3b2; --border:#262b34; --accent:#5b8ef0; --bar:#3f5f9e; --dim:#3a414d; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }}
.wrap {{ max-width:960px; margin:0 auto; padding:0 20px; }}
header {{ padding:56px 0 28px; }}
h1 {{ font-size:40px; margin:0; letter-spacing:-0.02em; }}
.lede {{ font-size:19px; color:var(--muted); max-width:720px; margin:10px 0 18px; }}
.links a {{ display:inline-block; margin-right:10px; padding:8px 14px; border:1px solid var(--border);
  border-radius:8px; color:var(--text); text-decoration:none; background:var(--panel); }}
.links a.primary {{ background:var(--accent); border-color:var(--accent); color:#fff; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:8px 0 8px; }}
.card {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:16px; }}
.big {{ font:600 22px var(--mono); }}
.small {{ color:var(--muted); font-size:13px; }}
section {{ background:var(--panel); border:1px solid var(--border); border-radius:12px; padding:22px; margin:18px 0; }}
h2 {{ margin:0 0 6px; font-size:21px; }}
p.note {{ color:var(--muted); margin:4px 0 14px; }}
.chart {{ width:100%; height:auto; }}
.chart .label, .chart .value {{ font:13px var(--mono); fill:var(--text); }}
.chart .bar {{ fill:var(--bar); }} .chart .accent {{ fill:var(--accent); }} .chart .muted {{ fill:var(--dim); }}
img {{ width:100%; border-radius:8px; border:1px solid var(--border); }}
table {{ border-collapse:collapse; width:100%; font:14px var(--mono); }}
td, th {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--border); }}
ul {{ padding-left:20px; }} code {{ font-family:var(--mono); font-size:14px; }}
footer {{ color:var(--muted); font-size:13px; padding:20px 0 48px; }}
@media (max-width:760px) {{ .cards {{ grid-template-columns:repeat(2,1fr); }} h1 {{ font-size:32px; }} }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>nanoinfer</h1>
  <p class="lede">A C++20 runtime that runs GPT-2 small on a CPU with no ML framework: hand-written
  AVX2 GEMM and GEMV kernels, int8 weights, a KV cache and a pinned thread pool, measured against
  the machine's roofline and against PyTorch.</p>
  <div class="links"><a class="primary" href="{REPO}">Source on GitHub</a>
  <a href="{REPO}#build-and-run">Build and run</a><a href="{REPO}/tree/main/results">Raw results</a></div>
</header>
<div class="cards">{card_html}</div>

<section>
  <h2>Decode: tokens per second</h2>
  <p class="note">128 greedy tokens after a 32-token prompt, same prompt and thread count for every
  system; each system at its fastest thread count, PyTorch also at the same counts.</p>
  {bar_chart(decode_rows, "decode tokens per second")}
</section>

<section>
  <h2>GEMM 1024 x 3072 x 768: one optimization at a time</h2>
  <p class="note">GFLOPS, one core for steps 1-5 (peak {peak["theoretical_gflops"]:.1f} GFLOPS at
  {peak["clock_ghz_under_fma_load"]:.2f} GHz), then the AVX2 kernel split across cores.</p>
  {bar_chart(gemm_rows, "GEMM GFLOPS")}
</section>

<section>
  <h2>Roofline</h2>
  <p class="note">Achieved GFLOPS against arithmetic intensity, with ceilings from the measured
  FMA peak and read bandwidth ({m5["measured_bandwidth_gb_per_second"]:.1f} GB/s).
  Decode is bandwidth-bound, which is why int8 weights help.</p>
  <img src="roofline.png" alt="Roofline plot of the GEMM steps, kernels and decode">
  <table><tr><th>Decode</th><th>tok/s</th><th>bandwidth ceiling</th><th>of ceiling</th></tr>{ceiling_rows}</table>
</section>

<section>
  <h2>Quality: WikiText-2 perplexity</h2>
  <p class="note">First {perplexity["slice"]["tokens"]:,} tokens of the test split, window
  {perplexity["slice"]["window"]}, stride {perplexity["slice"]["stride"]}, one scoring rule for both
  runtimes.</p>
  <table><tr><th>Configuration</th><th>Perplexity</th></tr>{ppl_rows}</table>
</section>

<section>
  <h2>Live demo (runs locally)</h2>
  <p class="note">The demo streams tokens from the native runtime, so it runs on your machine:
  <code>uv run python -m demo.server</code>, then open <code>http://127.0.0.1:8008</code>.</p>
  <img src="demo.jpg" alt="Screenshot of the local demo streaming generated text with live statistics">
</section>

<section>
  <h2>How it is tested</h2>
  <ul>
    <li>Every optimized kernel has a scalar reference twin and differential tests on awkward shapes.</li>
    <li>Logits within 1e-3 of PyTorch and identical greedy tokens; hidden states checked against a float64 oracle.</li>
    <li>Cached decode equals full recomputation bit for bit; output is byte-identical at any thread count.</li>
    <li>ASan, fatal UBSan and TSan in CI on Linux and Windows; malformed-file and corruption tests.</li>
  </ul>
</section>

<footer>Measured on {html.escape(str(machine.get("cpu", "the benchmark machine")))},
{html.escape(str(machine.get("physical_cores", "")))} cores, one DDR5 channel, on AC power.
Generated by <code>scripts/build_site.py</code> from <code>results/</code>.</footer>
</div>
</body>
</html>
"""
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.html").write_text(page, encoding="utf-8")
    for image in ("roofline.png", "demo.jpg"):
        shutil.copyfile(ROOT / "docs" / image, args.output / image)
    print(f"wrote {args.output / 'index.html'}")


if __name__ == "__main__":
    main()
