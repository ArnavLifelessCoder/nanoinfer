"""Capture native correctness-driver output with a bounded execution time."""

import subprocess
import tempfile
from pathlib import Path


def capture(command, *, timeout=60):
    directory = Path(__file__).resolve().parents[1] / "build" / "native-capture"
    directory.mkdir(parents=True, exist_ok=True)
    # Wait for the process, not pipe EOF: a descendant may retain inherited
    # output handles after the process being tested has already exited.
    with (
        tempfile.TemporaryFile(dir=directory) as stdout,
        tempfile.TemporaryFile(dir=directory) as stderr,
    ):
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout)
        stdout.seek(0)
        stderr.seek(0)
        result.stdout = stdout.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        result.stderr = stderr.read().decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        return result
