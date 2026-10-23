# M7 review packet

Benchmarks, all on AC power with background CPU load around 5% (recorded in
each JSON), Windows Balanced plan, turbo left enabled.

| Benchmark (spec) | Result file | Notes |
|---|---|---|
| GEMM 768^3 and 1024x3072x768 per optimization step, GFLOPS and % of peak | [m4-gemm](../results/m4-gemm.md) (steps 1-5, one core), [m7-kernels](../results/m7-kernels.md) (step 6, 1-16 threads) | Peak factors recorded: cores, clock under FMA load, 2 FMA pipes, 8 lanes, 2 flops |
| GEMV 768->3072 and 768->50257, f32 vs int8, GB/s and % of measured bandwidth | [m5-bandwidth](../results/m5-bandwidth.json), [m5.md](../results/m5.md) | L3-resident shapes exceed DRAM bandwidth; explained in [m5-review](m5-review.md) |
| Prefill 128 / 512 tokens | [m7-e2e](../results/m7-e2e.json), [m7.md](../results/m7.md) | Runtime f32 and int8 and PyTorch |
| Decode 128 after a 32-token prompt, tok/s, ms/token p50/p99 | same | Each step timed alone |
| Thread scaling | same, 1/2/4/8/16 threads for all three systems | |
| PyTorch CPU baseline, same prompts and thread count | `bench/pytorch_baseline.py` | float32, `torch.inference_mode()`, `torch.set_num_threads`, eager attention, HF KV cache |

Decode ceilings are computed in `scripts/plot_roofline.py` from the measured
read bandwidth and the weight bytes read per token (all matrices, the tied
embedding and one position row): 59.9 tok/s for f32 and 212 tok/s for int8.
The best measured decode reaches 70% and 38% of them.

An earlier run started on battery and ended on AC; it was discarded and the
benchmarks were rerun on AC. `bench/run_m7.py` now refuses to run on battery.
