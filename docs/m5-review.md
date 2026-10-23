# M5 review packet

Code: branch `m5`. The final benchmark results were produced later, on AC power
with the M7 drivers (which add power-state recording), and are committed on `m7`.

| Acceptance criterion | Evidence | Status |
|---|---|---|
| TSan clean | CI `tsan` job: every test labeled `concurrency` (unit tests including the thread pool, `thread_invariance`, `golden_int8`, `perplexity_consistency`) under ThreadSanitizer with `halt_on_error=1`; green on `m5` and later branches | Met |
| Thread scaling curve (1, 2, 4, ... physical cores) committed | [m5.md](../results/m5.md), [m5-scaling.json](../results/m5-scaling.json); runtime and PyTorch curves in [m7.md](../results/m7.md) | Met |
| Golden outputs still match with any thread count | `thread_invariance`: byte-identical captures and tokens for 1-4 threads and prompts up to 40 tokens; `Gemv.ResultIsBitIdenticalForEveryThreadCount` and `Gemv.ParallelGemmIsBitIdenticalForEveryThreadCount` | Met |
| GEMV GB/s against measured bandwidth (STREAM-like triad) | [m5-bandwidth.json](../results/m5-bandwidth.json): triad and a read-only stream at 1-16 threads; GEMV f32 and int8 as % of the read-only bandwidth | Met |

## Design

- `ThreadPool`: a fixed set of workers, one pinned per physical core (the
  caller is participant 0), static contiguous partitioning, first exception
  rethrown after all chunks finish. Idle workers spin about 100 us on an atomic
  generation counter before sleeping on a condition variable, because decode
  issues about a hundred short parallel regions per token; the wake-up protocol
  uses sequentially consistent operations so no wake-up is lost.
- Parallel GEMM splits B packing by panel and computation into
  MC x (8 x NR) tiles; each tile is computed by one thread in a fixed order.
- The decode GEMV sums fixed 32-row blocks into partial rows and adds them in
  block order; the transposed (logits) GEMV is one dot product per output.
- Optimized attention splits work into head x 16-row units and uses the AVX2
  dot and axpy primitives; `ops::ref::cached_attention` is its reference twin.

## Bandwidth

Decode reads weights only, so the read-only stream (29.6 GB/s) is its ceiling;
the triad (16.5 GB/s, STREAM byte counting) also pays for writes on this
single-channel machine. The 768->3072 GEMV (9.4 MB of weights) and the int8
embedding (43 MB) fit in the 64 MB L3, so their rates exceed DRAM bandwidth;
the f32 embedding (154 MB) streams from DRAM at 41.4 GB/s at 16 threads, above
the read benchmark, which suggests the simple read loop does not fully
saturate the memory system either. Both numbers are reported as measured.
