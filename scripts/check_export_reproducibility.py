"""Re-export the pinned GPT-2 revision and compare weights and golden arrays."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    metadata = json.loads((ROOT / "golden/manifest.json").read_text(encoding="utf-8"))
    output = ROOT / "build/recheck"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "python.export",
            "--revision",
            metadata["revision"],
            "--output",
            str(output / "gpt2.bin"),
            "--golden",
            str(output / "golden"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert not result.stderr.strip(), result.stderr
    assert digest(ROOT / "models/gpt2.bin") == digest(output / "gpt2.bin")
    assert json.loads(result.stdout) == json.loads(
        (ROOT / "models/gpt2.json").read_text(encoding="utf-8")
    )
    repeated = json.loads((output / "golden/manifest.json").read_text(encoding="utf-8"))
    assert repeated == metadata
    for prompt in metadata["prompts"]:
        with (
            np.load(ROOT / "golden" / prompt["file"], allow_pickle=False) as expected,
            np.load(output / "golden" / prompt["file"], allow_pickle=False) as actual,
        ):
            assert set(actual.files) == set(expected.files)
            for key in expected.files:
                np.testing.assert_array_equal(actual[key], expected[key])
    report = (
        "PASS repeated pinned-revision export: model SHA256 and every golden array "
        "are identical; stderr is empty\n"
    )
    (ROOT / "results/m1-export-reproducibility.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
