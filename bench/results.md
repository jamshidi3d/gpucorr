# gpucorr benchmark

GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8.0 GiB) · CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (32 threads, TreeCorr OpenMP) · precision: mixed · N_data = 10000 · periodic box L = 1000, 50 log bins 50–150, 50 φ bins

## Wall time per term on the GPU (s)

| N_rand | DD | DR | RR | DDD | DRR | RDD | RRR | total | peak VRAM (MiB) | batches |
|---|---|---|---|---|---|---|---|---|---|---|
| 10000 | 0.00 | 0.01 | 0.00 | 0.25 | 0.76 | 0.76 | 0.26 | 2.04 | 102 | 6 |
| 20000 | 0.03 | 0.11 | 0.07 | 0.42 | 1.64 | 1.12 | 0.77 | 4.16 | 253 | 6 |
| 50000 | 0.03 | 0.25 | 0.06 | 0.30 | 4.80 | 1.99 | 3.53 | 10.98 | 1086 | 6 |
| 100000 | 0.03 | 0.18 | 0.18 | 0.26 | 11.74 | 3.40 | 16.25 | 32.05 | 3707 | 6 |
| 200000 | 0.03 | 0.25 | 0.49 | 0.26 | 30.92 | 6.13 | 98.28 | 136.36 | 7784 | 12 |
| 500000 | 0.01 | 0.25 | 2.86 | 0.25 | 142.58 | 14.37 | 1409.86 | 1570.18 | 4339 | 53 |

## Total wall time (all 7 terms, s) and GPU speed-up

| N_rand | GPU exact | CPU TreeCorr default (NNN multipole) | speed-up | CPU TreeCorr triangle (default bin_slop) | speed-up | CPU TreeCorr exact (bin_slop=0) | speed-up |
|---|---|---|---|---|---|---|---|
| 10000 | 2.04 | 27.87 | 14× | 247.11 | 121× | 243.05 | 119× |
| 20000 | 4.16 | 42.82 | 10× | 551.31 | 133× | 550.21 | 132× |
| 50000 | 10.98 | 100.21 | 9× | ~1.59e+03 (extrap.) | 145× (extrap.) | ~1.62e+03 (extrap.) | 148× (extrap.) |
| 100000 | 32.05 | 213.17 | 7× | ~3.55e+03 (extrap.) | 111× (extrap.) | ~3.67e+03 (extrap.) | 115× (extrap.) |
| 200000 | 136.36 | 545.35 | 4× | ~7.93e+03 (extrap.) | 58× (extrap.) | ~8.3e+03 (extrap.) | 61× (extrap.) |
| 500000 | 1570.18 | ~1.14e+03 (extrap.) | 1× (extrap.) | ~2.29e+04 (extrap.) | 15× (extrap.) | ~2.45e+04 (extrap.) | 16× (extrap.) |

Fitted scaling t ∝ N_rand^α: gpu: α = 1.65, cpu_default: α = 0.99, cpu_triangle: α = 1.16, cpu_exact: α = 1.18

## Bounded-VRAM run

N_rand = 50000 with `max_vram_gb: 0.5` → 13 batches (vs 6 uncapped), peak 331 MiB (vs 1049 MiB); counts identical: **True**, max |Δzeta| = 0.0e+00; time 10.68 s vs 10.90 s.

