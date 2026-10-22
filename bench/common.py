"""Shared helpers for the benchmark drivers."""

import json
import statistics
import subprocess
import sys
from pathlib import Path

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
SUFFIX = ".exe" if sys.platform == "win32" else ""

# A fixed, original passage. Every runtime and PyTorch measurement uses the
# same token IDs, cut from this text repeated.
PROMPT_TEXT = (
    "The lighthouse keeper wrote down the weather every evening: the direction of the "
    "wind, the height of the waves and the colour of the sky over the harbour. Years "
    "later the notebooks were the only record of how the coast had changed, and a "
    "student who found them in an attic spent a winter turning them into tables. "
)


def prompt_ids(count):
    encoding = tiktoken.get_encoding("gpt2")
    ids = []
    while len(ids) < count:
        ids += encoding.encode_ordinary(PROMPT_TEXT)
    return ids[:count]


def write_ids(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(map(str, prompt_ids(count))), encoding="utf-8")
    return path


def summary(values):
    ordered = sorted(values)
    p99 = ordered[min(len(ordered) - 1, round(0.99 * (len(ordered) - 1)))]
    return dict(
        median=statistics.median(values),
        minimum=ordered[0],
        maximum=ordered[-1],
        p99=p99,
        samples=len(values),
    )


def run_google_benchmark(executable, repetitions, pin_core=2, extra=()):
    command = [
        str(executable),
        f"--pin-core={pin_core}",
        f"--benchmark_repetitions={repetitions}",
        "--benchmark_format=json",
        *extra,
    ]
    report = json.loads(subprocess.run(command, capture_output=True, text=True, check=True).stdout)
    rows = {}
    for row in report["benchmarks"]:
        if row.get("run_type") == "iteration" and not row.get("error_occurred"):
            rows.setdefault(row["run_name"].removesuffix("/real_time"), []).append(row)
    return report, rows


def sanitize(value):
    if isinstance(value, dict) and "host_name" in value:
        value = {**value, "host_name": "<host>"}
    text = json.dumps(value)
    for old, new in ((str(ROOT), "<project>"), (str(Path.home()), "<user>")):
        for form in (old, old.replace("\\", "\\\\"), old.replace("\\", "/")):
            text = text.replace(form, new)
    return json.loads(text)
