# gpucorr — CUDA backend for TreeCorr NN / NNN

`gpucorr` runs TreeCorr's count–count correlations, the 2-point **NN** (`nn_file_name`) and
the 3-point **NNN with `bin_type: LogSAS`** (`nnn_file_name`), on an NVIDIA GPU.

You use the same config file for both backends and choose between them with one key.

| `backend` | what runs | counting |
|---|---|---|
| `cpu` (default) | TreeCorr itself (`treecorr.corr2` / `corr3`), untouched | TreeCorr's tree algorithm with your `bin_slop` and `algo` settings |
| `cuda` | gpucorr on the GPU | **exact**: every pair or triangle in range is counted. This equals TreeCorr with `bin_slop=0` and `algo='triangle'`. |

Only the pair and triangle accumulation is replaced. Everything else is TreeCorr's own code:
- reading the catalogs and config;
- `finalize`;
- `calculateXi` (Landy–Szalay) and `calculateZeta` (DDD − RDD + DRR − RRR);
- the shot-noise variance;
- `write`.

The output files therefore have exactly TreeCorr's columns and format.

## Install

```bash
git clone https://github.com/<user>/gpucorr && cd gpucorr
pip install -e ".[cuda12]"     # CUDA 12 driver (e.g. Google Colab, where CuPy is preinstalled anyway)
pip install -e ".[cuda13]"     # CUDA 13 driver
```

You don't need a C++/CUDA compiler. The kernels (`gpucorr/kernels.cu`) are compiled at
runtime by NVRTC through CuPy, and the compiled code is cached after the first run (about 3 s).

## Run

```bash
python run_corr.py configs/p2pcfconfig.yaml                  # backend from the file (cpu)
python run_corr.py configs/p3pcfconfig.yaml backend=cuda     # override on the command line
python run_corr.py configs/p3pcfconfig.yaml backend=cuda max_vram_gb=12 nnn_file_name=3pcf_gpu.txt
gpucorr configs/p3pcfconfig.yaml backend=cuda                # same, via the installed command
```

Command-line `key=value` overrides work exactly as for TreeCorr's `corr2` / `corr3`.

### Config keys (added at the bottom of both yaml files)

```yaml
backend: "cpu"            # "cpu" | "cuda"
# cuda-only, all optional:
# gpu_device: 0           # GPU index
# max_vram_gb: 12         # cap on the GPU memory gpucorr may use (default: half of the free physical VRAM)
# gpu_precision: "mixed"  # "mixed" (default, fast) | "double"
# gpu_guard: 1.0e-5       # fp32 -> fp64 recheck band for phi bin edges (radians); rarely changed
# gpu_method: "exact"     # 3PCF method: "exact" (default) | "legendre"
# gpu_lmax: 10            # highest multipole for "legendre" (<= 63)
```

## 3PCF method: `exact` (default) or `legendre`

Switch methods with one line, e.g. `gpu_method: "legendre"` in `configs/p3pcfconfig.yaml`, or on the
command line: `python run_corr.py configs/p3pcfconfig.yaml backend=cuda gpu_method=legendre gpu_lmax=10`.

**`legendre`** is the spherical-harmonic estimator of Slepian & Eisenstein (2015).
- For every centre and radial bin it accumulates a_ℓm = Σ w Y_ℓm(r̂).
- By the addition theorem, Σ over neighbour pairs of w P_ℓ(cos φ) becomes
  Σ_m a_ℓm(r₂) a*_ℓm(r₃), which is one complex matrix product per ℓ over all centres (cuBLAS).
- The cost grows as N·n·ℓ_max² instead of N·n².

What it gives you:
- **Exact Legendre multipoles** of the triangle counts for every ℓ ≤ `gpu_lmax`, written to
  `<nnn_file_name>_legendre.txt`. That file holds DDD_ℓ, RRR_ℓ, DRR_ℓ, RDD_ℓ for each (d2, d3),
  plus Z_ℓ and ζ_ℓ = Z_ℓ / (RRR₀ f), so that ζ(φ) ≈ Σ (2ℓ+1) ζ_ℓ P_ℓ(cos φ).
- **The usual TreeCorr output file.** Its φ-bins are reconstructed from the truncated
  multipole series, which is an approximation that converges as ℓ_max grows.
  - The counts are symmetrised over d2↔d3, because the method has no 3D orientation.
  - meand2/meand3 are exact per (d2, d3) bin.
  - meanphi and meand1 are bin-centre values.

Use `legendre` if your analysis works with the multipoles ζ_ℓ, which is the usual choice for 3D
3PCFs. Use `exact` if you need the φ-binned counts themselves.

The comparison with `exact` (accuracy versus ℓ_max, and timing) is in
`report/gpucorr_vs_treecorr.pdf` (section 5; LaTeX source `report/gpucorr_vs_treecorr.tex`, rebuilt by `python report/make_report_tex.py`).

These keys are removed before the config reaches TreeCorr, so `backend: cpu` behaves exactly
like plain TreeCorr. With `backend: cuda`, `bin_slop`, `angle_slop`, `algo`, `max_n` and
`brute` are ignored, because counting is always exact.

## Python API

```python
import treecorr, gpucorr
data = treecorr.Catalog('input/data.data', x_col=1, y_col=2, z_col=3, w_col=4)
rand = treecorr.Catalog('input/random.txt', x_col=1, y_col=2, z_col=3)
kw = dict(min_sep=50, max_sep=150, nbins=50, nphi_bins=50, bin_type='LogSAS', period=1000)
opts = gpucorr.GPUOptions(gpu_precision='mixed', max_vram_gb=12)

ddd = gpucorr.GPUNNNCorrelation(gpu_opts=opts, **kw); ddd.process(data, metric='Periodic')
rrr = gpucorr.GPUNNNCorrelation(gpu_opts=opts, **kw); rrr.process(rand, metric='Periodic')
drr = gpucorr.GPUNNNCorrelation(gpu_opts=opts, **kw); drr.process(data, rand, ordered=False, metric='Periodic')
rdd = gpucorr.GPUNNNCorrelation(gpu_opts=opts, **kw); rdd.process(rand, data, ordered=False, metric='Periodic')
zeta, varzeta = ddd.calculateZeta(rrr=rrr, drr=drr, rdd=rdd)
ddd.write('3pcf_gpu.txt', rrr=rrr, drr=drr, rdd=rdd)
```

`GPUNNCorrelation` and `GPUNNNCorrelation` are subclasses of `treecorr.NNCorrelation` and
`treecorr.NNNCorrelation`, so every TreeCorr method and attribute works on them.

Two convenience entry points are also provided:
- `gpucorr.corr2(config)` and `gpucorr.corr3(config)`, the drop-in counterparts of TreeCorr's.
- `gpucorr.run_config(config)`, which dispatches on `backend` and on which output file is set.

## What "exact" means, and how it compares with TreeCorr

**Algorithm.** A periodic (or bounding-box) cell list enumerates candidate neighbours of every
point, and each candidate is tested with TreeCorr's own leaf-level formulas, taken from
`Metric.h`, `BinType.h`, `Corr2.cpp` and `Corr3.cpp` of TreeCorr 5.1:
- the min-image periodic wrap;
- the range test `min_sep ≤ d < max_sep`;
- the log-bin index `int((log d − log min_sep)/bin_size)`;
- `cos φ = (d2² + d3² − d1²)/(2 d2 d3)`, with `φ < max_phi`;
- the 2↔3 labelling from `CCW(P1,P3,P2)`.

No cell is ever accepted as a whole, and there is no multipole truncation.

- **`gpu_precision: mixed`** (default):
  - Geometry runs in fp32. Any triangle whose φ lies within 1e-5 rad of a bin edge, or which is
    nearly degenerate or has an ambiguous orientation, is recomputed in fp64 with TreeCorr's
    exact formula.
  - Bin assignment and counts (`ntri`, `npairs`) are therefore exact. Weights are exact for
    unit weights, and so is zeta, which depends only on the weights.
  - The mean columns (`meand1`, `meanphi`, …) agree with TreeCorr to ~1e-6 relative.
- **`gpu_precision: double`**:
  - Every triangle is evaluated in fp64 and accumulated in exact 64-bit fixed point.
  - Means agree with TreeCorr to 1e-12–1e-15. This mode is about 20× slower on consumer GPUs.
- **2-point** always runs fully in fp64. Counts are exact and means agree with TreeCorr to ~1e-15.

**One caveat specific to 3D periodic boxes.** TreeCorr decides which side is d2 and which is
d3 with `cross(r13, r12) · p1 < 0`. That is a dot product with the absolute position of a
vertex, a "line-of-sight" convention meant for the sky. In the auto-correlation it evaluates
this test once, at whichever vertex the tree traversal reaches first, and reuses it for all
three rotations of the triangle.

For a triangle that straddles the periodic boundary, the sign can depend on that vertex. The
d2↔d3 assignment of such triangles then depends on the tree traversal order and cannot be
reproduced point by point. gpucorr always applies the test at the triangle's own P1.

Consequences:
- Euclidean metric: GPU and TreeCorr agree bin for bin.
- Periodic metric: the following agree exactly:
  - totals;
  - all φ-marginals;
  - the histogram summed over (d2,d3)+(d3,d2).
- Per bin, a fraction of order 1e-3 of the triangles sits in the mirror bin (d3,d2) instead.
  See `tests/validation_report.md` for the measured numbers.
- An independent brute-force implementation that applies the same per-P1 rule matches the GPU
  bin for bin in both metrics.

## Memory: any GPU size

- Catalog positions stay on the GPU, which takes about 40 bytes per point.
- The large object is the per-centre neighbour list: 24 B per neighbour in mixed mode, 4 B in
  double mode.
- Centres are processed in batches sized to the free VRAM, or to `max_vram_gb`. Results
  accumulate into fixed 50×50×50 fp64/uint64 arrays.
- The same run therefore works on a 4 GB laptop GPU, on a 12 GB Colab budget or on an 80 GB
  A100. A smaller budget only means more batches, with identical numbers; the benchmark
  verifies this.
- Each kernel launch is sized to roughly 0.25 s, so a GPU that also drives a display does not
  hit the Windows/WDDM watchdog.

## Validation and benchmark

```bash
python tests/validate_vs_treecorr.py --ndata 2000 --nrand 4000   # -> tests/validation_report.md
python bench/benchmark.py --nrand 10000,20000,50000,100000,200000,500000 --cap 900
                                                                  # -> bench/results.md / .csv
```

`notebooks/colab_gpucorr.ipynb` runs both on Colab.

## Limitations

- Only 3D `x, y, z` catalogs (`coords='3d'`) are supported, with the `Euclidean` or `Periodic`
  metric. NN uses `bin_type: Log`; NNN uses `bin_type: LogSAS`.
- No patches (`npatch > 1`) and no three-catalog `process_cross`. The corr2/corr3 workflow
  (DD, DR, RR / DDD, DRR, RDD, RRR) never needs either.
- For jackknife or bootstrap covariances, use `backend: cpu`.
