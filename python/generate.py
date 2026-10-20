"""Tokenize a prompt with GPT-2 BPE, call gpt2rt, and decode the generated IDs."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

import tiktoken  # noqa: E402


def generate(
    executable: Path,
    model: Path,
    prompt: str,
    *,
    max_new: int = 128,
    temperature: float = 0,
    top_k: int = 0,
    top_p: float = 1,
    seed: int = 42,
) -> str:
    encoding = tiktoken.get_encoding("gpt2")
    ids = encoding.encode_ordinary(prompt)
    if not ids:
        raise ValueError("prompt must encode to at least one token")
    work = ROOT / "build" / "generation"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prompt-", dir=work) as directory:
        path = Path(directory) / "ids.txt"
        path.write_text(" ".join(map(str, ids)), encoding="utf-8")
        result = subprocess.run(
            [
                str(executable.resolve()),
                "--model",
                str(model.resolve()),
                "--prompt-ids",
                str(path),
                "--max-new",
                str(max_new),
                "--temp",
                str(temperature),
                "--top-k",
                str(top_k),
                "--top-p",
                str(top_p),
                "--seed",
                str(seed),
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    output = json.loads(result.stdout)
    if output["prompt_ids"] != ids or len(output["generated_ids"]) != max_new:
        raise ValueError("runtime returned inconsistent token IDs")
    return encoding.decode(ids + output["generated_ids"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new", type=int, default=128)
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        print(
            generate(
                args.executable,
                args.model,
                args.prompt,
                max_new=args.max_new,
                temperature=args.temp,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
        )
    except subprocess.CalledProcessError as error:
        parser.exit(1, error.stderr)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
