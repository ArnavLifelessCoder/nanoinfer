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
pretrained model; generated tiny fixtures live in the build directory. M2 will
add the C++ forward comparison to these fixtures.
