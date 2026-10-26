"""The demo's streamed generation must equal gpt2rt's normal JSON output."""

import argparse
import json
from pathlib import Path

import tiktoken

from demo.server import command, generate_events, validate
from tests.native_process import capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    encoding = tiktoken.get_encoding("gpt2")
    # Single characters map to GPT-2 IDs below 97, the tiny model vocabulary.
    for request in [
        {"prompt": "!", "max_new": 12, "temperature": 0.0, "threads": 1},
        {"prompt": "#", "max_new": 20, "temperature": 0.9, "top_k": 30, "seed": 7, "threads": 3},
    ]:
        settings = validate(request | {"model": "tiny"}, {"tiny": args.model}, 16)
        events = list(
            generate_events(args.executable.resolve(), args.model.resolve(), settings, encoding)
        )
        kinds = [kind for kind, _ in events]
        assert kinds[0] == "prompt" and kinds[1] == "prefill" and kinds[-1] == "done", kinds
        assert "error" not in kinds, events
        streamed = [data["id"] for kind, data in events if kind == "token"]
        assert len(streamed) == settings["max_new"]
        assert all(data["ms"] >= 0 for kind, data in events if kind in ("prefill", "token"))
        # Same settings without --stream: the final JSON must list the same tokens.
        args.directory.mkdir(parents=True, exist_ok=True)
        ids_path = args.directory / "ids.txt"
        ids_path.write_text(" ".join(map(str, encoding.encode_ordinary(settings["prompt"]))))
        arguments = command(args.executable.resolve(), args.model.resolve(), ids_path, settings)
        plain = capture(arguments[: arguments.index("--stream")])
        assert plain.returncode == 0, plain.stderr
        assert json.loads(plain.stdout)["generated_ids"] == streamed
        print(f"PASS streamed {len(streamed)} tokens equal the JSON output: {streamed[:6]}...")


if __name__ == "__main__":
    main()
