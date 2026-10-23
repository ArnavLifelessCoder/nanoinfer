# M6 review packet

Review `git diff m5...m6`. Design and results: [quantization.md](quantization.md).

| Acceptance criterion | Evidence | Status |
|---|---|---|
| Quantize/dequantize round-trip error bounds tested | `tests/test_quantize.py`: error at most half a step for every element, both axes and several shapes; group maximum maps to 127; zero groups; invalid shapes. Kernel twins: `tests/test_quantized.cpp` | Met |
| Perplexity table: PyTorch f32, runtime f32, runtime int8 | [m6-perplexity.md](../results/m6-perplexity.md) from `python/perplexity.py` and `scripts/perplexity_table.py`; same slice, token IDs, windows and scoring rule | Met |
| Token agreement over the golden prompts | [m6-int8-golden.json](../results/m6-int8-golden.json): 68.75% mean greedy agreement with float32 PyTorch | Met |

Perplexity: PyTorch float32 26.0162, runtime float32 26.0162, runtime int8
26.2467 (+0.2305).

The int8 kernels are also checked against their reference twin (the scalar
products on the exactly dequantized weights) on the tiny model in CI and on
the pretrained model here. On the pretrained model's large residual channel the
blocked kernels differ from the twin by float32 summation-order noise, as in
M4, so the check applies M4's float64 rule: an out-of-bound tensor passes when
its error against a float64 run of the int8 weights (`python/oracle.py
--quant int8`) is within the larger of the twin's and PyTorch float32's own
float64 error, plus 1e-4. All greedy continuations match the twin exactly.

## Local validation

Debug: 20/20 CTest. Local MSVC ASan: one full run that spanned an overnight
sleep timed out in `model_schema` and had a DLL initialization failure
(`0xC000026B`) starting `golden_negative`; no dump was captured, so under the M3
gate it is not evidence either way. Rerunning the three affected tests passed
3/3. The CI ASan jobs on both platforms are the gate.

## Reproduce

```powershell
python -m python.export --revision 607a30d783dfa663caf39e06633721c8d4cfcd7e --quant int8 --output models/gpt2-int8.bin
python -m python.oracle --golden golden --quant int8
python -m tests.golden_int8 --executable build/release/gpt2opt.exe --model models/gpt2-int8.bin `
  --golden golden --directory build/release/int8-pretrained --report results/m6-int8-golden.json `
  --reference-kernel blocked --threads 8 --timeout 3600
python -m python.perplexity --revision b08601e04326c79dfdd32d625aee71d232d685c3 `
  --ids build/wikitext2-slice-ids.txt --report results/m6-perplexity-pytorch.json
python scripts/perplexity_table.py
```

`uv sync --group bench` installs `pyarrow` for the dataset download.
