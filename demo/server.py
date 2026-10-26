"""Local demo: a web UI that streams tokens from the native runtime.

    uv run python -m demo.server            # then open http://127.0.0.1:8008

The server listens on 127.0.0.1 only. It tokenizes the prompt with tiktoken,
runs `gpt2rt --stream 1` with the chosen model, kernel threads and sampling
settings, and forwards each token to the browser as a server-sent event,
decoded incrementally (a GPT-2 token can end in the middle of a UTF-8
character). Only model files under models/ that the server lists are accepted,
and arguments go to the runtime as a list, never through a shell.
"""

import argparse
import codecs
import json
import os
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(ROOT / ".cache" / "tiktoken"))

import tiktoken  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
SUFFIX = ".exe" if sys.platform == "win32" else ""
MODELS = {
    "f32": ("gpt2.bin", "GPT-2 small, float32 (498 MB)"),
    "int8": ("gpt2-int8.bin", "GPT-2 small, int8 weights (143 MB)"),
}
CONTEXT = 1024
CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class TokenDecoder:
    """Turns token IDs into text pieces, holding back incomplete UTF-8 bytes."""

    def __init__(self, encoding):
        self.encoding = encoding
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def piece(self, token_id):
        return self.decoder.decode(self.encoding.decode_single_token_bytes(token_id))

    def flush(self):
        return self.decoder.decode(b"", final=True)


def validate(request, models, max_threads):
    """Checked generation settings, or ValueError with a message for the user."""

    def number(name, kind, low, high, default):
        value = request.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        value = kind(value)
        if not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
        return value

    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be non-empty text")
    model = request.get("model", "int8")
    if model not in models:
        raise ValueError(f"model must be one of: {', '.join(sorted(models))}")
    return dict(
        prompt=prompt,
        model=model,
        threads=number("threads", int, 1, max_threads, min(4, max_threads)),
        max_new=number("max_new", int, 1, 512, 64),
        temperature=number("temperature", float, 0.0, 5.0, 0.8),
        top_k=number("top_k", int, 0, 50257, 40),
        top_p=number("top_p", float, 0.01, 1.0, 0.95),
        seed=number("seed", int, 0, 2**32 - 1, 42),
    )


def command(executable, model_path, ids_path, settings):
    return [
        str(executable),
        "--model",
        str(model_path),
        "--prompt-ids",
        str(ids_path),
        "--max-new",
        str(settings["max_new"]),
        "--temp",
        repr(settings["temperature"]),
        "--top-k",
        str(settings["top_k"]),
        "--top-p",
        repr(settings["top_p"]),
        "--seed",
        str(settings["seed"]),
        "--threads",
        str(settings["threads"]),
        "--stream",
        "1",
    ]


def generate_events(executable, model_path, settings, encoding):
    """Yield (event, data) pairs for one generation, ending with done or error."""
    ids = encoding.encode_ordinary(settings["prompt"])
    if len(ids) + settings["max_new"] > CONTEXT:
        yield "error", dict(message=f"prompt ({len(ids)} tokens) plus new tokens exceeds {CONTEXT}")
        return
    yield "prompt", dict(tokens=len(ids))
    with tempfile.TemporaryDirectory(prefix="nanoinfer-demo-") as directory:
        ids_path = Path(directory) / "ids.txt"
        ids_path.write_text(" ".join(map(str, ids)), encoding="utf-8")
        process = subprocess.Popen(
            command(executable, model_path, ids_path, settings),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        decoder = TokenDecoder(encoding)
        try:
            for line in process.stdout:
                event = json.loads(line)
                kind = event.pop("event")
                if kind == "token":
                    event["text"] = decoder.piece(event["id"])
                elif kind == "done":
                    event["text"] = decoder.flush()
                yield kind, event
            process.wait(timeout=30)
            if process.returncode != 0:
                yield "error", dict(message=process.stderr.read().strip() or "runtime failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


class Handler(BaseHTTPRequestHandler):
    server_version = "nanoinfer-demo"

    def log_message(self, format, *args):  # Quiet: one line per generation instead.
        pass

    def send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/info":
            return self.send_json(200, self.server.info)
        name = "index.html" if self.path in ("/", "/index.html") else self.path.lstrip("/")
        path = (STATIC / name).resolve()
        if path.parent != STATIC or not path.is_file() or path.suffix not in CONTENT_TYPES:
            return self.send_json(404, dict(error="not found"))
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES[path.suffix] + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/generate":
            return self.send_json(404, dict(error="not found"))
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 65536:
                raise ValueError("request body must be 1 to 65536 bytes")
            settings = validate(
                json.loads(self.rfile.read(length)), self.server.models, self.server.max_threads
            )
        except (ValueError, json.JSONDecodeError) as error:
            return self.send_json(400, dict(error=str(error)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        model_path = self.server.models[settings["model"]]
        try:
            for kind, data in generate_events(
                self.server.executable, model_path, settings, self.server.encoding
            ):
                self.wfile.write(f"event: {kind}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # The browser stopped the stream; generate_events kills the runtime.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=ROOT / "build" / "release")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--port", type=int, default=8008)
    args = parser.parse_args()
    executable = (args.build / f"gpt2rt{SUFFIX}").resolve()
    if not executable.is_file():
        raise SystemExit(f"build the runtime first: {executable} is missing")
    models = {
        key: (args.models / filename).resolve()
        for key, (filename, _) in MODELS.items()
        if (args.models / filename).is_file()
    }
    if not models:
        raise SystemExit(f"export a model first: no {', '.join(f for f, _ in MODELS.values())}")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.executable = executable
    server.models = models
    server.max_threads = os.cpu_count() or 1
    server.encoding = tiktoken.get_encoding("gpt2")
    results = ROOT / "results" / "m7-e2e.json"
    baseline = {}
    if results.exists():
        for row in json.loads(results.read_text(encoding="utf-8"))["results"]["PyTorch f32"]:
            baseline[row["threads"]] = row["decode"]["tokens_per_second"]["median"]
    server.info = dict(
        models=[dict(key=key, label=MODELS[key][1]) for key in models],
        max_threads=server.max_threads,
        context=CONTEXT,
        pytorch_decode_tokens_per_second=baseline,
    )
    print(f"nanoinfer demo on http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
