# gpucorr

**gpucorr is a CUDA implementation of two established estimators of the count–count correlation
functions:** the 2-point **NN** (ξ) and the 3-point **NNN / `LogSAS`** (ζ(d₂, d₃, φ)).

| `gpu_method` | counterpart of | what it computes |
|---|---|---|
| **`exact`** (default) | **[TreeCorr](https://github.com/rmjarvis/TreeCorr)** (Jarvis et al.) | The **exact counterpart of TreeCorr**. Every pair and triangle is counted with TreeCorr's own leaf-level formulas: the same distances, periodic wrapping, log/φ binning and d₂/d₃ orientation. The result equals TreeCorr with `bin_slop=0` (and `algo='triangle'` for NNN), with no tree approximation. |
| **`legendre`** | **Slepian & Eisenstein (2015)** | The **multipole counterpart of the Slepian–Eisenstein estimator**. The Legendre multipoles ζ_ℓ(d₂, d₃) come from spherical-harmonic sums a_ℓm, at a cost of N·n·ℓ²_max instead of N·n². They are exact for ℓ ≤ ℓ_max. φ-binned values are reconstructed from them. |

It reads the same config files as TreeCorr, and one key selects the backend:

```yaml
backend: "cuda"          # or "cpu" = plain TreeCorr
gpu_method: "exact"      # or "legendre"
gpu_lmax: 10             # for gpu_method: legendre
```

Only the pair and triangle accumulation moves to the GPU. Catalogue reading, `calculateXi`,
`calculateZeta`, the variances and `write` are TreeCorr's own code, so the output files are
identical in format. The Legendre mode also writes the multipoles to `*_legendre.txt`.

## Results

**Test case:** 10,000 data points (all of them, in every run) and N_rand randoms in a periodic box
of side 1000, with 50 log bins from 50 to 150 and 50 φ bins. The GPU is an RTX 4070 Laptop GPU
(8 GB); the CPU runs used TreeCorr 5.1.4 with OpenMP on 32 threads. Details, figures and every
number are in [`report/gpucorr_vs_treecorr.pdf`](report/gpucorr_vs_treecorr.pdf).

### Speed

The table gives the wall time for the full estimator set (DD, DR, RR, DDD, DRR, RDD, RRR). CPU runs
were capped at 10 min per size; entries marked ~ are per-term power-law extrapolations beyond the
cap. For triangle and exact these rest on two measured sizes, so they are order-of-magnitude
estimates.

| N_rand | GPU exact | GPU legendre ℓ_max=10 | GPU legendre ℓ_max=49 | CPU TreeCorr default (`multipole`)¹ | CPU TreeCorr `triangle` | CPU TreeCorr exact (`bin_slop=0`) |
|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 2.0 s | 0.9 s | 8.4 s | 28 s | 4.1 min | 4.1 min |
| 20,000 | 4.2 s | 1.2 s | 13 s | 43 s | 9.2 min | 9.2 min |
| 50,000 | 11 s | 2.6 s | 30 s | 100 s | ~37 min | ~36 min |
| 100,000 | 32 s | 5.7 s | 64 s | 3.6 min | ~2.1 h | ~2.0 h |
| 200,000 | 2.3 min | 15 s | 2.6 min | 9.1 min | ~8.4 h | ~7.7 h |
| 500,000 | **26 min** | **69 s** | 11 min | ~25 min | ~58 h | ~49 h |

¹ Inaccurate for 3D catalogues; see the next table.

GPU speed-ups at 500k randoms:

| GPU method | vs CPU `triangle` | vs CPU exact |
|---|---:|---:|
| exact | ~130× | ~110× |
| legendre, ℓ_max=10 | ~3,000× | ~2,500× |

### Precision

| comparison | metric / precision | result |
|---|---|---|
| **GPU exact vs TreeCorr exact** (`bin_slop=0`, real catalogues, all 7 terms) | Euclidean | **0 bins differ**, Δξ = 0, Δζ = 0 |
| | Periodic | totals, φ-marginals and d₂↔d₃-symmetrised counts identical; ≤ 0.06 % of triangle rotations sit in the mirror bin (d₃, d₂)² |
| **GPU exact vs independent brute force** (bin means d̄₁, d̄₂, φ̄) | mixed | counts identical; means ≤ 1e-6 (rel.) |
| | double | counts identical; means ≤ 2e-12 (rel.) |
| **GPU legendre vs independent brute force** (multipoles, ℓ ≤ 12) | mixed / double | 3e-7 / **4e-16** |
| **GPU legendre vs GPU exact**, φ-binned ζ | ℓ_max = 10 / 49 | rms Δζ = 0.87σ / 0.45σ (σ = shot noise of ζ); the difference is high-ℓ noise filtering, not bias |
| **GPU legendre vs GPU exact**, multipoles ζ_ℓ | | monopole identical; higher ℓ differ only through the exact method's φ-binning (4× finer bins → 4× smaller difference) |
| CPU TreeCorr default (2PCF, `bin_slop`≈1) vs exact | | max Δξ = 5e-3 |
| CPU TreeCorr `triangle` (default `bin_slop`) vs exact, 3D NNN | | median 0.17 % per bin |
| **CPU TreeCorr default (`multipole`) vs exact, 3D NNN** | | **median 32 % per bin; φ-marginal off by up to 20×** (2D: 1.7 %) |

² TreeCorr orients 3D triangles by a dot product with the absolute position of the vertex its tree
traversal reaches first. For triangles crossing the periodic boundary this makes its d₂/d₃
labelling depend on traversal order. gpucorr always uses P₁. See Sec. 4 of the report.

> **Note on TreeCorr's default 3PCF algorithm.** For 3D (x, y, z) catalogues, TreeCorr 5.1's
> default LogSAS algorithm (`multipole`) does not reproduce the opening-angle distribution: it
> returns a flat φ-distribution for uniform randoms instead of sin φ (Sec. 4.1 of the report).
> Use `algo: triangle` on the CPU, or gpucorr.

### Other properties

- **Any GPU size.** Work is batched to the available VRAM, or to `max_vram_gb` (e.g. 12 GB on
  Colab). A 0.5 GB cap gave identical counts at the same speed.
- **No compiler needed.** The CUDA kernels are compiled at runtime by CuPy/NVRTC.

## Install

```bash
git clone https://github.com/jamshidi3d/gpucorr && cd gpucorr
pip install -e ".[cuda12]"      # CUDA 12 (Colab); use ".[cuda13]" for a CUDA 13 driver
```

## Quick start

Put your catalogues in `input/` (format described in [`input/README.md`](input/README.md)), then:

```bash
python run_corr.py configs/p2pcfconfig.yaml backend=cuda                 # xi
python run_corr.py configs/p3pcfconfig.yaml backend=cuda                 # zeta, exact
python run_corr.py configs/p3pcfconfig.yaml backend=cuda gpu_method=legendre gpu_lmax=10
python run_corr.py configs/p3pcfconfig.yaml                              # backend from the file (cpu = TreeCorr)
```

From Python:

```python
import treecorr, gpucorr
opts = gpucorr.GPUOptions(gpu_method='exact', max_vram_gb=12)
ddd = gpucorr.GPUNNNCorrelation(min_sep=50, max_sep=150, nbins=50, nphi_bins=50,
                                bin_type='LogSAS', period=1000, gpu_opts=opts)
ddd.process(data_cat, metric='Periodic')      # a treecorr.NNNCorrelation in every other respect
```

The full guide (config keys, the Python API, precision modes, the Legendre output format,
memory and limitations) is in **[USAGE.md](USAGE.md)**. Colab users should start from
[`notebooks/colab_gpucorr.ipynb`](notebooks/colab_gpucorr.ipynb).

## Repository layout

```
gpucorr/          the package: CUDA kernels (kernels.cu), engines, TreeCorr subclasses, CLI
configs/          example corr2/corr3 config files with the backend / gpu_* keys
run_corr.py       command-line entry point (same as `gpucorr` / `python -m gpucorr`)
tests/            consistency checks against TreeCorr and independent brute-force references
bench/            benchmark script and results (results.md / .csv / .json)
report/           LaTeX report (PDF + source), its generator and figure data
notebooks/        Google Colab notebook
input/            your catalogues (not distributed)
```

## Validation and benchmarks

```bash
python tests/validate_vs_treecorr.py      # GPU vs TreeCorr exact + NumPy brute force -> tests/validation_report.md
python tests/check_legendre.py            # Legendre multipoles vs brute force
python tests/compare_legendre.py          # Legendre vs exact: accuracy vs lmax, timing
python bench/benchmark.py                 # GPU vs TreeCorr timing table -> bench/results.md
python report/make_report_tex.py          # rebuild report/gpucorr_vs_treecorr.pdf (needs LaTeX)
```

## Limitations

- Supported: 3D x, y, z catalogues; `Euclidean` or `Periodic` metric; NN with `bin_type: Log`;
  NNN with `bin_type: LogSAS`.
- Not supported: patches (`npatch > 1`, jackknife covariances) and three-catalogue NNN cross
  correlations. Use `backend: cpu` for those.

## References

- M. Jarvis, G. Bernstein & B. Jain (2004), *The skewness of the aperture mass statistic*, MNRAS
  352, 338 (TreeCorr).
- Z. Slepian & D. J. Eisenstein (2015), *Computing the three-point correlation function of galaxies
  in O(N²) time*, MNRAS 454, 4142.

## License

BSD 2-Clause (see [LICENSE](LICENSE)). gpucorr reproduces the leaf-level conventions of
TreeCorr, which is also BSD-licensed.
