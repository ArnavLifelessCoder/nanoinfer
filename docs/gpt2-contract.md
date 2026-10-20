# GPT-2 source checks

The exporter targets GPT-2 small and checks its configuration at runtime:
12 blocks, width 768, 12 attention heads, vocabulary 50257 and context 1024.
It requires `gelu_new`, LayerNorm epsilon `1e-5`, ordinary scaled self-attention
and a four-times-width MLP.

Sources inspected for the implementation:

- [GPT-2 model configuration](https://huggingface.co/gpt2/blob/main/config.json)
- [Transformer blocks and forward pass, transformers v4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/gpt2/modeling_gpt2.py)
- [Conv1D storage and multiplication](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/pytorch_utils.py)

`GPT2Block` applies LayerNorm before each sublayer and adds the residual after
each sublayer. Attention projects fused QKV, divides attention scores by the
square root of the head dimension and applies a causal mask. `ln_f` follows the
last block. `Conv1D` allocates `[nx, nf]` weights and computes an addmm directly
with those weights, so the exporter deliberately does not transpose them.

Golden fixtures use float32 CPU evaluation, eager attention, one PyTorch thread,
no KV cache, and no dropout. Hooks capture each complete block output before the
next LayerNorm. The final LayerNorm output is saved separately; this avoids
mistaking the last entry of Hugging Face `hidden_states` for a raw block output.
Each fixture stores logits for every prompt position and exactly 32 greedy
continuation token IDs, without stopping early at EOS.

Local acceptance uses pretrained GPT-2. CI uses a reproducibly seeded tiny GPT-2
with two blocks, width 16, two heads, vocabulary 97 and context 64. Tiny-model
weights are regenerated, never committed. The committed golden set is from the
pretrained model; generated tiny fixtures live in the build directory. M2's
`golden_tiny` CTest runs the native comparison, including 32 generated tokens.

M2 uses scalar float32 accumulators and tanh GELU. The cubic term is evaluated
before multiplying by `0.044715`, matching the expression in Hugging Face
`NewGELUActivation`; changing the grouping changes float32 rounding.
Attention includes keys 0 through the query position, with independent heads.
Generation recomputes the prefix and its absolute positions; it projects only
the last position to logits when selecting the next token.

The owner approved an M2 numerical clarification: for every hidden-state
element, including final LayerNorm, require
`abs(actual - expected) <= 1e-4 + 1e-6 * abs(expected)`. Logits still require
strict maximum absolute error below `1e-3`, and greedy IDs must match exactly.
This addresses float32 spacing at large pretrained hidden activations; it is
not a tensor-wide relative allowance. The written project spec is unchanged.

The owner withdrew the additional strict `1e-4` requirement for elements with
`abs(torch) < 1024`. The reviewer traced the larger errors in channels 138, 447
and 373 to float32 accumulation rather than an implementation error. Those
elements still use the approved combined bound. Their subset maximum remains
visible in the statistics; it is not a separate acceptance gate.

`scripts/hidden_error_stats.py` records every block and final LayerNorm for each
prompt: absolute error, the below-1024 subset maximum, relative error for
`abs(torch) >= 1`, the count above absolute `1e-4`, and worst-error coordinates
with the reference magnitude. Coordinates are zero-based. A separate entry
identifies the worst element under the combined tolerance.

`golden_negative` runs the normal reference and two separately compiled
test-only variants, substituting exact erf GELU and LayerNorm epsilon `1e-6`.
A fixed seed-42 tiny model uses MLP input weights multiplied by 20 and output
weights by 5 to exercise GELU away from its nearly linear region. PyTorch
produces fresh goldens with the correct activation and epsilon. The normal
forward must pass; both altered forwards must fail the same numerical gate
on every prompt. Test switches are private to the variant libraries and are
absent from the normal library and CLI.

For M4, compare the complete optimized forward against this scalar reference
as well as PyTorch. Layer 11 is already close to the combined tolerance bound,
so PyTorch-only checks do not adequately isolate numerical drift introduced
by each optimization.
