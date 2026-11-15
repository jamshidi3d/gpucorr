# Quick start

## 1. Put your catalogues in `input/`

The example configs read two whitespace-separated ASCII files, in Cartesian coordinates:

```
input/data.data     x y z w        (w = weight, column 4)
input/random.txt    x y z
```

Other file names or column layouts: see [Configuration](configuration.md#input-catalogues).

## 2. Compute ξ (2-point) on the GPU

```bash
python run_corr.py configs/p2pcfconfig.yaml backend=cuda
```

This runs DD, DR and RR on the GPU, forms the Landy–Szalay ξ with TreeCorr's `calculateXi`, and
writes `2pcf_periodic.txt` (the `nn_file_name` in the config).

## 3. Compute ζ (3-point) on the GPU

The exact method counts every triangle into (d₂, d₃, φ) bins:
```bash
python run_corr.py configs/p3pcfconfig.yaml backend=cuda
```

The Legendre multipole method (Slepian & Eisenstein) is much faster at high density:
```bash
python run_corr.py configs/p3pcfconfig.yaml backend=cuda gpu_method=legendre gpu_lmax=10
```

Both run DDD, DRR, RDD and RRR, form the full estimator ζ = (DDD − RDD + DRR − RRR)/RRR, and
write `3pcf_periodic.txt`. The Legendre run also writes `3pcf_periodic_legendre.txt` with the
multipoles ζ_ℓ.

## 4. Make it permanent

Instead of command-line overrides, set the keys at the bottom of the config file:

```yaml
backend: "cuda"
gpu_method: "legendre"     # or "exact"
gpu_lmax: 10
max_vram_gb: 12            # optional: cap GPU memory, e.g. on Colab
```

then run `python run_corr.py configs/p3pcfconfig.yaml`.

## 5. Compare with TreeCorr on the CPU

The same file with `backend=cpu` runs unmodified TreeCorr:

```bash
python run_corr.py configs/p2pcfconfig.yaml backend=cpu nn_file_name=2pcf_cpu.txt
```

For a like-for-like CPU reference of the exact method, add `bin_slop=0`. For the 3-point
function also add `algo=triangle`: TreeCorr's default `multipole` algorithm is not correct for
3D catalogues (see [Methods](methods.md#a-warning-about-treecorrs-default-3pcf-algorithm)).
Exact TreeCorr is slow, so use a subsample.

## Next steps

- Every config key: [Configuration reference](configuration.md)
- Calling gpucorr from Python: [Python API](python-api.md)
- Choosing between `exact` and `legendre`, `mixed` and `double`: [Methods](methods.md)
