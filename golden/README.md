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
