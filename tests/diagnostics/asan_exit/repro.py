"""Run a model-free exit probe; stop and capture the first failure, never retry."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--procdump", type=Path, required=True)
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--check-errors", action="store_true")
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    records = []
    cases = (
        [("heap-overflow", "heap-buffer-overflow"), ("use-after-free", "heap-use-after-free")]
        if args.check_errors
        else [(None, None)] * args.count
    )
    for index, (mode, diagnostic) in enumerate(cases):
        expected = None if args.check_errors else (2 if index % 2 else 0)
        arguments = [mode] if mode else (["exit-two"] if expected else [])
        command = [str(args.executable.resolve()), *arguments]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record = dict(index=index, pid=process.pid, expected_exit=expected, diagnostic=diagnostic)
        try:
            stdout, stderr = process.communicate(timeout=10)
            record.update(
                exit=process.returncode,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            if diagnostic:
                record["passed"] = (
                    process.returncode != 0
                    and f"ERROR: AddressSanitizer: {diagnostic}".encode() in stdout + stderr
                )
            else:
                record["passed"] = (
                    process.returncode == expected
                    and stdout.replace(b"\r\n", b"\n") == b"probe complete\n"
                    and not stderr
                )
        except subprocess.TimeoutExpired:
            dump = args.directory / f"probe-{index}-{process.pid}.dmp"
            with (args.directory / f"probe-{index}-capture.txt").open("wb") as log:
                captured = subprocess.run(
                    [str(args.procdump), "-accepteula", "-ma", str(process.pid), str(dump)],
                    stdout=log,
                    stderr=log,
                    timeout=30,
                )
            record.update(
                passed=False,
                timeout=True,
                dump=dump.name,
                capture_exit=captured.returncode,
                dump_exists=dump.exists(),
            )
            process.kill()
        records.append(record)
        if not record["passed"]:
            print(f"FAIL iteration {index}: {record}", flush=True)
            break
    report = dict(
        asan_options=os.environ.get("ASAN_OPTIONS", ""),
        requested=len(cases),
        completed=len(records),
        passed=len(records) == len(cases) and all(record["passed"] for record in records),
        records=records,
    )
    (args.directory / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed {len(records)} attempts; passed={report['passed']}", flush=True)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
