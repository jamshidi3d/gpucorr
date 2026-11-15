# gpucorr

**gpucorr is an exact CUDA backend for [TreeCorr](https://github.com/rmjarvis/TreeCorr)'s
count–count correlation functions.** It computes the 2-point **NN** (ξ) and the 3-point
**NNN / `LogSAS`** (ζ(d₂, d₃, φ)) correlations on an NVIDIA GPU. It reads the same config files as
TreeCorr, and one key selects the backend:

```yaml
backend: "cuda"          # or "cpu" = plain TreeCorr
gpu_method: "exact"      # or "legendre" (Slepian & Eisenstein multipoles, optional)
```

Only the pair and triangle accumulation moves to the GPU. Catalogue reading, `calculateXi`,
`calculateZeta`, the variances and `write` are TreeCorr's own code, so the output files are
identical in format.

## Highlights

These figures come from 10,000 data points and 500,000 randoms in a periodic box, 50 log bins
from 50 to 150 and 50 φ bins, on an RTX 4070 Laptop GPU. The full results are in
[`report/gpucorr_vs_treecorr.pdf`](report/gpucorr_vs_treecorr.pdf).

| | wall time, all 7 terms (DD…RRR) |
|---|---|
| **gpucorr, exact** | **26 min** |
| TreeCorr, `triangle` algorithm (accurate CPU baseline) | ~58 h (extrapolated) |
| TreeCorr, `bin_slop=0` (exact CPU) | ~49 h (extrapolated) |
| **gpucorr, `legendre`, ℓ_max = 10** (3PCF only) | **66 s** |

- **Exact counting.**
  - Every pair and triangle is tested with TreeCorr's own leaf formulas.
  - Against TreeCorr with `bin_slop=0`, every term agrees bin for bin (Euclidean metric), with
    Δξ = Δζ = 0.
  - In the default `mixed` precision, counts, weights and ζ are exact and the mean columns agree
    to about 1e-6. `gpu_precision: double` brings the means to 1e-12–1e-15.
- **Legendre mode** (`gpu_method: legendre`).
  - It computes the Slepian & Eisenstein (2015) multipoles ζ_ℓ(d₂, d₃) exactly, verified to
    4e-16 against brute force.
  - At this density it is 24× faster than exact counting at ℓ_max = 10.
- **Any GPU size.** Work is batched to the available VRAM, or to `max_vram_gb` (e.g. 12 GB on
  Colab). Results don't depend on the batch size.
- **No compiler needed.** The CUDA kernels are compiled at runtime by CuPy/NVRTC.

> **Note on TreeCorr's default 3PCF algorithm.** For 3D (x, y, z) catalogues, TreeCorr 5.1's
> default LogSAS algorithm (`multipole`) does not reproduce the opening-angle distribution: it
> returns a flat φ-distribution for uniform randoms instead of sin φ (Sec. 4.1 of the report).
> Use `algo: triangle` on the CPU, or gpucorr.

## Install

```bash
git clone https://github.com/<user>/gpucorr && cd gpucorr
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
