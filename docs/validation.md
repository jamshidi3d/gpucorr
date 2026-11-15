# Validation and benchmarks

These scripts reproduce the consistency checks, the timing tables and the PDF report. The
scripts that use real catalogues need `input/data.data` and `input/random.txt` (see
[`input/README.md`](../input/README.md)). The others generate uniform random points.

| script | what it checks | output | typical time |
|---|---|---|---|
| `tests/quick_check.py N METRIC` | GPU vs TreeCorr exact (`bin_slop=0`) on N uniform points, both precisions | console | < 1 min for N ≈ 1500 |
| `tests/validate_vs_treecorr.py` | (1) GPU vs an independent NumPy brute force; (2) GPU vs TreeCorr exact on subsamples of the real catalogues, all 7 terms, ξ and ζ; (3) TreeCorr `multipole` vs `triangle` in 2D and 3D | `tests/validation_report.md/.json` | ~5 min |
| `tests/check_legendre.py` | Legendre multipoles vs a NumPy brute force of Σ P_ℓ(cos φ) | console | < 1 min |
| `tests/compare_legendre.py` | Legendre vs exact: φ-binned and multipole accuracy vs ℓ_max; timing scan | `report/legendre_results.json`, figures | ~10 min accuracy + ~40 min timing |
| `bench/benchmark.py` | GPU vs TreeCorr (default, triangle, exact) timing for several N_rand; bounded-VRAM test | `bench/results.md/.csv/.json` | ~1–2 h (CPU-bound) |
| `report/make_report_tex.py` | builds the LaTeX report from all of the above | `report/gpucorr_vs_treecorr.pdf` | ~1 min |

## Typical invocations

```bash
python tests/quick_check.py 1500 Euclidean
python tests/validate_vs_treecorr.py --ndata 2000 --nrand 4000
python tests/check_legendre.py
python tests/compare_legendre.py --nrand 50000 --timing 10000,50000,100000,500000
python tests/compare_legendre.py --timing ""          # accuracy only, keeps an earlier timing scan
python bench/benchmark.py --nrand 10000,20000,50000,100000,200000,500000 --cap 600
python bench/benchmark.py --skip-cpu --nrand 10000,100000      # GPU timings only
python report/make_report_tex.py
```

Useful `bench/benchmark.py` options:
- `--cap S`: wall-clock cap per CPU run (larger sizes are extrapolated);
- `--cpu-exact-max N`: largest N_rand at which exact TreeCorr is attempted;
- `--vram-test GB`: repeat one case under a memory cap and check the results are identical;
- `--resume FILE --redo N1,N2`: reuse finished rows of an earlier run.

## What "agree" means

- **Counts (`npairs`, `ntri`, `weight`):** must be identical bin for bin against TreeCorr exact
  in the Euclidean metric. In the periodic metric, the d₂ ↔ d₃-symmetrised counts must be
  identical; see [Methods](methods.md#periodic-boxes-and-triangle-orientation).
- **ξ, ζ:** identical, since they are functions of the counts.
- **Means:** ~1e-6 (mixed) or ~1e-12 (double), relative.
- **Legendre multipoles:** ~3e-7 (mixed) or ~4e-16 (double) against brute force.

The brute-force references in the tests share no code with the GPU kernels. They apply
TreeCorr's leaf formulas directly to every pair and triple.
