# Pretrained correctness fixtures

`manifest.json` records the exact model revision, library versions, prompt text,
token IDs and execution settings. Each `prompt_N.npz` contains:

- `input_ids`: int32 prompt tokens.
- `logits`: float32 logits at every prompt position.
- `layer_0` through `layer_11`: complete block outputs after both residual additions.
- `final_ln`: output after the final LayerNorm.
- `greedy_ids`: exactly 32 continuation IDs, without an EOS early stop.

These files contain reference outputs, not model parameters. Regenerate with
`python -m python.export --output models/gpt2.bin --golden golden`. The C++ forward
comparison will be added in M2. M1 tests the exporter capture against direct
PyTorch execution and validates every exported weight through the C++ loader.

The committed fixtures were produced with PyTorch **2.11.0+cu128**, using
float32 CPU tensors, eager attention and one CPU thread. The local venv reuses
that installed wheel; it was not a clean CPU-wheel `uv sync --locked` environment.
CI uses the locked CPU wheel. The exact versions and pinned model revision are
recorded in `manifest.json`.

Run `python scripts/check_export_reproducibility.py` to re-export that pinned
revision under `build/recheck/` and compare the model SHA-256 and every golden
array exactly. Its report is saved to `results/m1-export-reproducibility.txt`.
