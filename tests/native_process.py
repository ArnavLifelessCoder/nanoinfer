"""Capture native correctness-driver output with a bounded execution time."""

import subprocess
import tempfile
import os
import time
from pathlib import Path


def _capture_timeout(process, directory):
    procdump = os.environ.get("NANOINFER_PROCDUMP")
    if not procdump:
        return
    stem = directory / f"timeout-{process.pid}-{time.time_ns()}"
    with stem.with_suffix(".txt").open("wb") as log:
        try:
            subprocess.run(
                [procdump, "-accepteula", "-ma", str(process.pid), str(stem.with_suffix(".dmp"))],
                stdout=log,
                stderr=log,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            log.write(f"\nCapture failed: {error}\n".encode())


def capture(command, *, timeout=60):
    directory = Path(__file__).resolve().parents[1] / "build" / "native-capture"
    directory.mkdir(parents=True, exist_ok=True)
    # Wait for the process, not pipe EOF: a descendant may retain inherited
    # output handles after the process being tested has already exited.
    with (
        tempfile.TemporaryFile(dir=directory) as stdout,
        tempfile.TemporaryFile(dir=directory) as stderr,
    ):
        with subprocess.Popen(command, stdout=stdout, stderr=stderr) as process:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    _capture_timeout(process, directory)
                finally:
                    process.kill()
                    process.wait()
                raise
            result = subprocess.CompletedProcess(command, process.returncode)
        stdout.seek(0)
        stderr.seek(0)
        result.stdout = stdout.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        result.stderr = stderr.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return result
