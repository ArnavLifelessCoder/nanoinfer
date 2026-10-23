# int8 weight-only quantization

`python -m python.export --quant int8` writes a format version 2 file. Every
Conv1D matrix (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) and the
tied embedding `wte` are stored as int8 with one float32 scale per group of 32
values along the input dimension: axis 0 for the Conv1D `[in, out]` matrices,
axis 1 for `wte`, which the logits use as `[N, K]`. Position embeddings,
LayerNorm parameters and biases stay float32. Activations are float32
throughout.

Quantization is symmetric: `scale = max(abs(w)) / 127` per group, and
`q = round(w / scale)`, clipped to [-127, 127]; an all-zero group has scale 0.
Dequantization is `float(q) * scale`, so each element's error is at most half a
step (`tests/test_quantize.py` checks this bound and the edge cases). The
exported file is 142.6 MB against 497.8 MB for float32.

## Format version 2

A q8 entry has dtype 2, rank 2, and `group | axis << 16` in the entry's
reserved field. Its payload is the int8 values, zero padding to a 4-byte
boundary, then the float32 scales; the checksum covers all of it. The loader
rejects q8 in a version 1 file, a group of 0 or one that does not divide the
axis, a size that disagrees with the shape, nonzero padding, and negative or
non-finite scales. The GPT-2 schema check allows q8 only on the five matrices
above, each with its expected axis. `tests/quantized_model.py` covers each case
through the native loader. A file without q8 tensors is still written as
version 1, byte for byte.

## Kernels

- Decode GEMV, `[K, N]`: one partial row per quantization group. Inside a
  group the inner loop loads 8 int8 values, sign-extends and converts them to
  float, and FMAs them with `x_k`; the group's scales are applied once to the
  partial row, and partial rows are added in group order. About 1.125 bytes
  are read per weight instead of 4.
- Decode GEMV, `[N, K]` (logits): one dot product per output, scaled per group.
- Prefill GEMM: B panels are dequantized to float32 while they are packed, so
  the M4 micro-kernels and the M5 tiling are unchanged.
- Embedding lookup: the token's int8 row is dequantized.

Each kernel has a reference twin: the scalar `ops::ref::matmul` on the exactly
dequantized float32 weights. `tests/test_quantized.cpp` compares them on
awkward shapes, group sizes 8 to 64 and both layouts, and checks bit-identical
results across thread counts. The scalar reference forward (no int8 kernels)
dequantizes the weights once at load, which makes it the int8 forward's
reference twin; `tests/golden_int8.py` compares the int8 kernels with it on
every golden prompt.

## Results

[m6-perplexity.md](../results/m6-perplexity.md) and
[m6-perplexity.json](../results/m6-perplexity.json): the first 16,384 tokens of
the WikiText-2 raw test split (pinned dataset revision and file hash), window
1024, stride 512, 16,383 tokens scored. PyTorch float32 and the runtime use the
same token IDs (their hashes are checked) and the same scoring rule;
`tests/perplexity_consistency.py` shows the two code paths agree on the tiny
model to six decimals.

[m6-int8-golden.json](../results/m6-int8-golden.json) records greedy agreement
with the float32 PyTorch golden continuations: 68.75% over the three prompts
(7/32, 32/32 and 27/32). Short prompts are fragile here: once a token differs,
the rest of the continuation diverges, so the per-prompt numbers swing widely;
the perplexity is the better measure of quality.
