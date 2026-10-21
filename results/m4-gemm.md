Single thread, core 2, 7 repetitions, median [min-max] GFLOPS.
Peak 62.3 GFLOPS = 1 core x 1.95 GHz x 2 FMA x 8 lanes x 2 (measured FMA probe: 62.3 GFLOPS).

| Step | 768x768x768 | % peak | 1024x3072x768 | % peak | Speedup vs naive |
|---|---|---|---|---|---|
| 1. naive | 1.13 [1.11-1.13] | 1.8% | 0.20 [0.20-0.20] | 0.3% | 1.0x |
| 2. reordered | 21.53 [21.49-21.60] | 34.6% | 21.54 [21.46-21.56] | 34.6% | 108.0x |
| 3. blocked | 24.86 [24.52-25.02] | 39.9% | 21.09 [20.94-21.16] | 33.9% | 105.7x |
| 4. packed | 5.08 [5.06-5.09] | 8.1% | 5.05 [5.03-5.08] | 8.1% | 25.3x |
| 5. avx2 | 53.21 [51.96-53.89] | 85.4% | 51.62 [46.17-53.66] | 82.9% | 258.7x |
