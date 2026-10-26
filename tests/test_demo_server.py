import pytest
import tiktoken

from demo.server import TokenDecoder, command, validate

MODELS = {"int8": "models/gpt2-int8.bin", "f32": "models/gpt2.bin"}


def test_validate_applies_defaults_and_bounds():
    settings = validate({"prompt": "Hello"}, MODELS, 16)
    assert settings == dict(
        prompt="Hello",
        model="int8",
        threads=4,
        max_new=64,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        seed=42,
    )
    assert validate({"prompt": "x", "threads": 2}, MODELS, 2)["threads"] == 2


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        {"prompt": "   "},
        {"prompt": 7},
        {"prompt": "x", "model": "../secret"},
        {"prompt": "x", "threads": 0},
        {"prompt": "x", "threads": 17},
        {"prompt": "x", "max_new": 513},
        {"prompt": "x", "temperature": -1},
        {"prompt": "x", "top_p": 0},
        {"prompt": "x", "seed": -1},
        {"prompt": "x", "seed": True},
        {"prompt": "x", "top_k": "40"},
    ],
)
def test_validate_rejects_bad_requests(request_body):
    with pytest.raises(ValueError):
        validate(request_body, MODELS, 16)


def test_decoder_holds_back_partial_utf8_until_complete():
    encoding = tiktoken.get_encoding("gpt2")
    text = "café ☃ \U0001f600 done"
    ids = encoding.encode_ordinary(text)
    decoder = TokenDecoder(encoding)
    pieces = [decoder.piece(token) for token in ids] + [decoder.flush()]
    assert "".join(pieces) == text
    assert all("�" not in piece for piece in pieces)


def test_command_is_an_argument_list_with_stream_mode():
    settings = validate({"prompt": "x; rm -rf /", "temperature": 0.5}, MODELS, 16)
    arguments = command("gpt2rt", "model.bin", "ids.txt", settings)
    assert arguments[0] == "gpt2rt" and "x; rm -rf /" not in arguments
    assert arguments[arguments.index("--stream") + 1] == "1"
    assert arguments[arguments.index("--temp") + 1] == "0.5"
