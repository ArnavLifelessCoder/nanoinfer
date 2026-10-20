"""The wrapper only tokenizes, invokes the native process and decodes."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from python import generate


class Encoding:
    def encode_ordinary(self, text):
        assert text == "Hello \u03bb"
        return [15496, 7377, 119]

    def decode(self, ids):
        assert ids == [15496, 7377, 119, 42, 43]
        return "Hello \u03bb continued"


def test_wrapper_passes_ids_options_and_decodes_with_project_local_temporary_files(monkeypatch):
    def encoding(name):
        assert name == "gpt2"
        return Encoding()

    monkeypatch.setattr(generate.tiktoken, "get_encoding", encoding)
    paths = []

    def run(command, **kwargs):
        options = dict(zip(command[1::2], command[2::2], strict=True))
        ids = Path(options["--prompt-ids"])
        assert ids.is_relative_to(generate.ROOT / "build" / "generation")
        assert ids.read_text(encoding="utf-8") == "15496 7377 119"
        assert options["--model"] == str(Path("model.bin").resolve())
        assert options["--max-new"] == "2" and options["--temp"] == "0.8"
        assert options["--top-k"] == "7" and options["--top-p"] == "0.9"
        assert options["--seed"] == "1234"
        assert kwargs["check"] and kwargs["capture_output"]
        paths.append(ids)
        return SimpleNamespace(
            stdout=json.dumps({"prompt_ids": [15496, 7377, 119], "generated_ids": [42, 43]})
        )

    monkeypatch.setattr(generate.subprocess, "run", run)
    assert (
        generate.generate(
            Path("runtime.exe"),
            Path("model.bin"),
            "Hello \u03bb",
            max_new=2,
            temperature=0.8,
            top_k=7,
            top_p=0.9,
            seed=1234,
        )
        == "Hello \u03bb continued"
    )
    assert not paths[0].exists()


def test_empty_prompt_does_not_launch_runtime(monkeypatch):
    monkeypatch.setattr(
        generate.tiktoken, "get_encoding", lambda _: SimpleNamespace(encode_ordinary=lambda _: [])
    )
    with pytest.raises(ValueError, match="at least one token"):
        generate.generate(Path("missing"), Path("missing"), "")


def test_native_failure_propagates_and_cleans_prompt_file(monkeypatch):
    monkeypatch.setattr(generate.tiktoken, "get_encoding", lambda _: Encoding())
    paths = []

    def fail(command, **kwargs):
        paths.append(Path(command[command.index("--prompt-ids") + 1]))
        raise subprocess.CalledProcessError(1, command, stderr="generation error")

    monkeypatch.setattr(generate.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        generate.generate(Path("runtime"), Path("model"), "Hello \u03bb")
    assert not paths[0].exists()
