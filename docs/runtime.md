# Cached generation

M3 uses the same scalar operations as the M2 reference. Each `CachedTransformer`
owns a mapped model and preallocated per-layer K and V arrays with logical shape
`[ctx, heads, head_dim]`. Fused QKV projections retain the original `[Q, K, V]`
layout; cache writes copy only K and V. The arrays do not grow during decoding.
Instances own independent caches and support one sequence each, with one native
thread. Calls on the same instance must not overlap.

`prefill(ids)` resets the valid prefix and fills positions `0..n-1`.
`decode(id)` appends one position. Position embeddings start at the valid prefix
length, and attention includes all keys through the query's absolute position.
Only the valid prefix is exposed to attention, so resetting or prefilling a
shorter prompt cannot read stale cache entries. `reset()` invalidates the prefix
without clearing the allocated arrays.

Empty prompts, invalid token IDs and exhausted context are rejected before
mutating the valid prefix. An exception during computation invalidates the
cache, requiring another prefill. Decode before prefill is an error.
The default API returns logits; optional capture returns complete block states
and the final LayerNorm output for correctness checks.

## Sampling and CLI

`gpt2rt` accepts whitespace-separated unsigned token IDs and emits one JSON
object containing `prompt_ids` and `generated_ids`. Errors go to stderr with a
nonzero status. Windows uses wide-character entry points and filesystem paths.

```powershell
.\build\debug\gpt2rt.exe --model models/gpt2.bin --prompt-ids build/ids.txt --max-new 32 --temp 0.8 --top-k 40 --top-p 0.9 --seed 42
python -m python.generate --executable build/debug/gpt2rt.exe --model models/gpt2.bin --prompt "Hello" --max-new 32 --temp 0.8 --top-p 0.9 --seed 42
```

The Python wrapper uses tiktoken's `gpt2` encoding, invokes the native executable,
and decodes the combined prompt and continuation. Its temporary ID files and
tokenizer cache stay inside the project. It does not load torch or evaluate a
model. Special-token-looking strings in the prompt are encoded as ordinary text;
empty prompts are rejected.

Defaults are `max-new=128`, `temp=0`, `top-k=0`, `top-p=1`, `seed=42`.
Prompt plus continuation must fit the model's context. Generation emits exactly
the requested count and does not stop at EOS, matching the reference driver.
The last emitted token needs no decode when no further token will be sampled.
The correctness driver separately decodes it and checks that final position too.

- Temperature zero selects greedy argmax; ties select the smallest ID.
  Top-k and top-p have no effect in greedy mode.
- Positive temperature scales logits before a stable softmax computed in double
  precision. Logits and options must be finite. Temperature must be nonnegative.
- Top-k zero disables filtering; a positive value retains at most that many
  tokens, ordered by descending logit with smaller IDs first on ties.
- Top-p is in `(0,1]` and is applied after renormalizing the top-k distribution.
  It retains the smallest prefix reaching the threshold, including the token
  crossing it, with at least one token retained.
- Sampling uses `mt19937_64`, converts its high 53 bits to `[0,1)`, and searches
  the cumulative probability. This fixes the RNG-to-uniform mapping without
  relying on `std::discrete_distribution`. The same seed and inputs reproduce
  the sequence in the same build. Different math libraries can round softmax
  boundaries differently; cross-platform bitwise stochastic output is not a
  promised contract. Invalid calls and greedy calls consume no random draws.

## Correctness

The original `ReferenceTransformer` remains the full-prefix reference. Cache
tests require exact finite float equality, with no tolerance, for all vocabulary
logits, complete block states and final LayerNorm outputs at every checked
position. Random tiny-model sequences exercise the entire context with prefill
lengths 1, 7, 31, 63 and 64, reset, shorter reuse and independent instances.
Cached attention also has randomized differential tests with sequence lengths
and head dimensions 1, 7, 63 and 65.

`gpt2cachecheck`, built only with tests enabled, has the same positional arguments
as `gpt2ref`. It captures cached prefill outputs, then checks every generated
position against full recomputation, including the last emitted token. A mismatch
fails immediately; zero differences in its JSON are recorded only after exact
elementwise comparisons pass. `tests/golden_cached.py` checks the complete
position coverage and applies the approved PyTorch golden bounds to the
captures, plus exact 32-token greedy agreement. CI uses the exported tiny model.
Local acceptance also uses all three committed pretrained golden prompts.

The M2 hidden-state tolerance remains elementwise
`1e-4 + 1e-6 * abs(torch)`; logits remain strictly below `1e-3` maximum absolute
error. The separate strict small-value hidden gate remains withdrawn.
M4 must retain comparisons against both the scalar reference and PyTorch.
