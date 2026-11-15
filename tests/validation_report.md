# gpucorr vs TreeCorr — consistency report

TreeCorr 5.1.4, gpucorr 0.1.0, binning: 50 log bins 50.0–150.0, 50 phi bins [0, π].

## Part 1 — independent NumPy brute force vs GPU (auto NNN, uniform points)

| metric | precision | N | triangles (rotations) | bins differing | max abs Δntri | max rel Δ meand1 | Δ meanlogd1 | Δ meand2 | Δ meanphi | exact-path triplets | orientation-ambiguous rotations |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Euclidean | mixed | 2000 | 561876 | 0 | 0 | 9.36e-07 | 7.83e-07 | 4.07e-09 | 6.85e-06 | 193 | 0 |
| Euclidean | double | 2000 | 561876 | 0 | 0 | 2.04e-12 | 1.26e-12 | 8.90e-15 | 4.97e-12 | 561876 | 0 |
| Periodic | mixed | 2000 | 733344 | 0 | 0 | 1.04e-06 | 8.35e-07 | 4.66e-09 | 6.85e-06 | 268 | 45486 |
| Periodic | double | 2000 | 733344 | 0 | 0 | 2.04e-12 | 1.26e-12 | 8.90e-15 | 4.97e-12 | 733344 | 45486 |

## Part 2 — TreeCorr exact (bin_slop=0, algo='triangle') vs GPU, real catalogs (subsample: 2000 data, 4000 randoms)

### Euclidean

**2-point (NN)**

| term | max abs Δnpairs | max rel Δweight | max rel Δmeanr | max abs Δmeanlogr |
|---|---|---|---|---|
| DD | 0 | 0.00e+00 | 1.76e-15 | 6.22e-15 |
| DR | 0 | 0.00e+00 | 5.36e-15 | 2.04e-14 |
| RR | 0 | 0.00e+00 | 2.86e-15 | 1.33e-14 |

max |Δxi| GPU vs exact: **0.00e+00**; TreeCorr default (bin_slop) vs exact: 4.93e-03 (max rel ΔRR 2.58e-03).

**3-point (NNN, LogSAS)**

| term | total ntri exact | total ntri GPU | bins differing | Σ\|Δntri\| | fraction moved (d2↔d3) | bins differing after d2↔d3 symmetrisation | φ-marginal max Δ | max rel Δmeand1 (equal-count bins) |
|---|---|---|---|---|---|---|---|---|
| DDD | 564830 | 564830 | 0 | 0 | 0.00e+00 | 0 | 0 | 1.18e-06 |
| DRR | 6610871 | 6610871 | 0 | 0 | 0.00e+00 | 0 | 0 | 1.28e-06 |
| RDD | 3345174 | 3345174 | 0 | 0 | 0.00e+00 | 0 | 0 | 1.86e-06 |
| RRR | 4355147 | 4355147 | 0 | 0 | 0.00e+00 | 0 | 0 | 1.50e-06 |

zeta, full estimator (DDD−RDD+DRR−RRR)/RRR, rms over bins = 1.63:

* GPU vs TreeCorr exact: max |Δzeta| = **0.00e+00** over all bins (0.00e+00 over bins with RRR ≥ 100); with the counts symmetrised over d2↔d3 before forming zeta: **0.00e+00**.
* TreeCorr *default* (multipole) vs TreeCorr exact: max |Δzeta| = 8.89e+03 (symmetrised 1.83e+03); max rel ΔRRR = 1.37e+02.

Timings (s): DDD: CPU exact 6.19 / CPU default 0.73 / GPU 0.04, DRR: CPU exact 18.93 / CPU default 3.15 / GPU 0.14, RDD: CPU exact 18.01 / CPU default 2.56 / GPU 0.10, RRR: CPU exact 6.69 / CPU default 1.40 / GPU 0.06

### Periodic

**2-point (NN)**

| term | max abs Δnpairs | max rel Δweight | max rel Δmeanr | max abs Δmeanlogr |
|---|---|---|---|---|
| DD | 0 | 0.00e+00 | 1.39e-15 | 6.22e-15 |
| DR | 0 | 0.00e+00 | 5.86e-15 | 1.95e-14 |
| RR | 0 | 0.00e+00 | 3.80e-15 | 2.84e-14 |

max |Δxi| GPU vs exact: **0.00e+00**; TreeCorr default (bin_slop) vs exact: 3.40e-03 (max rel ΔRR 2.26e-03).

**3-point (NNN, LogSAS)**

| term | total ntri exact | total ntri GPU | bins differing | Σ\|Δntri\| | fraction moved (d2↔d3) | bins differing after d2↔d3 symmetrisation | φ-marginal max Δ | max rel Δmeand1 (equal-count bins) |
|---|---|---|---|---|---|---|---|---|
| DDD | 744773 | 744773 | 838 | 838 | 5.63e-04 | 0 | 0 | 3.72e-03 |
| DRR | 8951544 | 8951544 | 1232 | 1238 | 6.92e-05 | 0 | 0 | 1.21e-04 |
| RDD | 4468204 | 4468204 | 500 | 502 | 5.62e-05 | 0 | 0 | 1.80e-04 |
| RRR | 5984644 | 5984644 | 1142 | 1154 | 9.64e-05 | 0 | 0 | 1.27e-03 |

zeta, full estimator (DDD−RDD+DRR−RRR)/RRR, rms over bins = 1.5:

* GPU vs TreeCorr exact: max |Δzeta| = **8.00e+00** over all bins (8.00e-02 over bins with RRR ≥ 100); with the counts symmetrised over d2↔d3 before forming zeta: **0.00e+00**.
* TreeCorr *default* (multipole) vs TreeCorr exact: max |Δzeta| = 3.55e+04 (symmetrised 3.31e+03); max rel ΔRRR = 1.51e+02.

Timings (s): DDD: CPU exact 8.35 / CPU default 0.73 / GPU 0.19, DRR: CPU exact 26.55 / CPU default 3.17 / GPU 0.28, RDD: CPU exact 25.92 / CPU default 2.58 / GPU 0.11, RRR: CPU exact 9.87 / CPU default 1.46 / GPU 0.07

## Part 3 — TreeCorr default LogSAS algorithm (multipole) in 2D vs 3D

Uniform points, same binning, auto-correlation; bins with > 300 triangles.

| dim | N | reference | multipole: median rel. Δ | multipole: max rel. Δ | multipole: φ-marginal max rel. Δ | triangle (default bin_slop): median | max |
|---|---|---|---|---|---|---|---|
| 2D | 6000 | TreeCorr triangle | 1.69e-02 | 4.61e-01 | 1.18e-02 | 0.00e+00 | 0.00e+00 |
| 3D | 20000 | GPU exact | 3.19e-01 | 2.35e+01 | 1.98e+01 | 1.70e-03 | 3.22e-02 |

In 2D the multipole algorithm agrees with the triangle algorithm at the per-cent level. For 3D (x, y, z) catalogs it does not reproduce the opening-angle distribution, whereas the TreeCorr triangle algorithm and gpucorr agree.

