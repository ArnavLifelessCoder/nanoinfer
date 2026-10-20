"""Output capture must follow the parent process lifetime, not inherited pipe EOF."""

import subprocess
import sys
import time

import pytest

from tests.native_process import capture


def test_capture_preserves_stdout_stderr_and_nonzero_exit():
    result = capture(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            "import sys; print('héllo'); print('error', file=sys.stderr); sys.exit(7)",
        ]
    )
    assert result.returncode == 7
    assert result.stdout == "héllo\n"
    assert result.stderr == "error\n"


def test_capture_timeout_is_an_error():
    with pytest.raises(subprocess.TimeoutExpired):
        capture([sys.executable, "-c", "import time; time.sleep(10)"], timeout=0.1)


def test_capture_does_not_wait_for_descendant_to_close_inherited_output(tmp_path):
    release, done = tmp_path / "release", tmp_path / "done"
    descendant = """
import sys, time
from pathlib import Path
release, done = map(Path, sys.argv[1:])
deadline = time.monotonic() + 5
while not release.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
done.write_text("done", encoding="utf-8")
"""
    parent = """
import subprocess, sys
subprocess.Popen([sys.executable, "-c", sys.argv[1], *sys.argv[2:]],
                 stdout=sys.stdout, stderr=sys.stderr)
print("parent finished", flush=True)
"""
    try:
        result = capture([sys.executable, "-c", parent, descendant, str(release), str(done)])
        assert result.returncode == 0 and result.stdout == "parent finished\n"
        assert not done.exists(), "capture waited for a descendant's inherited output handle"
    finally:
        release.touch()
        deadline = time.monotonic() + 10
        while not done.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert done.exists(), "descendant did not finish cleanup"
