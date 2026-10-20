"""Audit the saved M1 manifests, golden arrays, and final test logs."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "results/m1-export-manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((ROOT / "models/gpt2.json").read_text(encoding="utf-8"))
    assert manifest == saved, "printed and saved exporter manifests differ"
    config = manifest["config"]
    assert config == dict(layers=12, d_model=768, heads=12, vocab=50257, ctx=1024)
    assert len(manifest["tensors"]) == 148
    metadata = json.loads((ROOT / "golden/manifest.json").read_text(encoding="utf-8"))
    assert metadata["revision"] == manifest["source"]["revision"]
    fixtures = []
    for prompt in metadata["prompts"]:
        path = ROOT / "golden" / prompt["file"]
        length = len(prompt["ids"])
        with np.load(path, allow_pickle=False) as arrays:
            expected = {"input_ids", "logits", "final_ln", "greedy_ids"}
            expected |= {f"layer_{index}" for index in range(config["layers"])}
            assert set(arrays.files) == expected
            np.testing.assert_array_equal(arrays["input_ids"], prompt["ids"])
            assert arrays["logits"].shape == (length, config["vocab"])
            assert arrays["final_ln"].shape == (length, config["d_model"])
            assert arrays["greedy_ids"].shape == (32,)
            assert np.all((arrays["greedy_ids"] >= 0) & (arrays["greedy_ids"] < config["vocab"]))
            for index in range(config["layers"]):
                assert arrays[f"layer_{index}"].shape == (length, config["d_model"])
            for key in expected - {"input_ids", "greedy_ids"}:
                assert arrays[key].dtype == np.float32 and np.isfinite(arrays[key]).all()
        fixtures.append(
            {
                "file": prompt["file"],
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "tokens": length,
            }
        )
    assert len(fixtures) == 3
    for preset in ("debug", "asan"):
        output = (ROOT / f"results/m1-{preset}-tests.txt").read_text(encoding="utf-8")
        assert "100% tests passed, 0 tests failed out of 4" in output
        output = (ROOT / f"results/m1-{preset}-gpt2.txt").read_text(encoding="utf-8")
        assert output.count("\nPASS ") == 148
        assert output.rstrip().endswith("PASS all 148 tensor shapes, checksums, offsets and sizes")
    assert "20 passed" in (ROOT / "results/m1-python-tests.txt").read_text(encoding="utf-8")
    summary = {
        "milestone": "M1",
        "pretrained_tensor_count": len(manifest["tensors"]),
        "ctest_entries_per_preset": 4,
        "cpp_unit_tests": 12,
        "python_tests": 20,
        "golden_files": fixtures,
        "golden_npz_bytes": sum(fixture["bytes"] for fixture in fixtures),
        "model_revision": metadata["revision"],
        "local_torch_build": metadata["torch"],
        "execution_device": "cpu",
        "ci_status": "not run; push awaits owner approval",
        "tsan": "not applicable; no runtime worker threads",
    }
    text = json.dumps(summary, indent=2) + "\n"
    (ROOT / "results/m1-summary.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
