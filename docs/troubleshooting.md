# Troubleshooting and FAQ

## Error messages

| message | cause | fix |
|---|---|---|
| `backend 'cuda' needs CuPy and an NVIDIA GPU` | CuPy not installed | `pip install -e ".[cuda12]"` (or `cuda13`, matching `nvidia-smi`) |
| `backend 'cuda': no usable CUDA device found` | no GPU visible: driver missing, or a Colab CPU runtime | install the NVIDIA driver; on Colab choose a GPU runtime |
| `CUDA driver version is insufficient` / `cudaErrorInsufficientDriver` | CuPy wheel built for a newer CUDA than the driver | use `cupy-cuda12x` with a CUDA 12 driver |
| `gpucorr currently supports 3d (x,y,z) catalogs` | catalogue given as ra/dec, or only x/y | supply `z_col`; spherical/2D catalogues are CPU-only for now |
| `gpucorr supports metric Euclidean or Periodic` | `metric: Arc/Rperp/Rlens` | use `backend: cpu` for those metrics |
| `gpucorr NN supports bin_type='Log' only` / `NNN … 'LogSAS' only` | other bin types | use `backend: cpu` |
| `catalogs with npatch > 1 are not supported yet` | jackknife patches requested | drop `npatch` or use `backend: cpu` |
| `three-catalog NNN cross-correlations are not implemented` | `file_name2` / `file_name3` 3pt crosses | use `backend: cpu` |
| `gpu_lmax must be in [0, 63]` | `gpu_lmax` too large | use ≤ 63; above ~30, exact counting is usually faster |
| `Periodic metric requires setting the period` | `metric: Periodic` without `period` | add `period:` (or `xperiod`/`yperiod`/`zperiod`) |
| `Invalid parameter …` (from TreeCorr) | misspelled key | check the spelling; gpucorr keys all start with `gpu_` (plus `backend`, `max_vram_gb`) |

## The run is unexpectedly slow

- **Windows, GPU at 100 % but low power draw:** GPU memory is being paged to system RAM. Close
  other GPU programs or set `max_vram_gb` lower. See
  [Memory and performance](memory-and-performance.md#windows-laptops-and-shared-gpus).
- **Laptop on battery:** the GPU is power-limited; plug in.
- **The first run of a session takes a few extra seconds:** that is kernel compilation, cached
  afterwards.
- **`gpu_precision: double` on a consumer GPU:** expected to be 10–20× slower. Use `mixed`
  unless you need 1e-12 means.

## FAQ

**Is `backend: cuda` an approximation of TreeCorr?**
No. The exact method counts every pair and triangle, which is what TreeCorr does with
`bin_slop=0`. TreeCorr's *default* settings are the approximate ones: the tree's `bin_slop` for
both functions, and in 3D the `multipole` algorithm (see
[Methods](methods.md#a-warning-about-treecorrs-default-3pcf-algorithm)).

**Why don't my periodic-box 3PCF bins match TreeCorr exactly?**
For triangles crossing the box boundary, TreeCorr's 3D orientation rule depends on its tree
traversal order. Only the d₂ ↔ d₃ labelling differs, and the symmetrised counts are identical.
See [Methods](methods.md#periodic-boxes-and-triangle-orientation).

**Should I use `exact` or `legendre`?**
If your analysis uses ζ_ℓ multipoles (common for 3D 3PCFs), use `legendre`: it is exact for
those and faster. If you need ζ in φ bins, or a bin-for-bin comparison with TreeCorr, use
`exact`.

**Does `max_vram_gb` change the result?**
No. It only changes how many batches the work is split into.

**Can I use weights?**
Yes, through `w_col`. With non-unit weights, mixed mode accumulates weight products in fp32
(≈1e-7 relative); use `double` if that matters.

**Can I run on AMD or Apple GPUs?**
Not at present; the kernels are CUDA and run through CuPy on NVIDIA hardware.

## Limitations

- GPU support covers:
  - 3D Cartesian catalogues;
  - the `Euclidean` and `Periodic` metrics;
  - NN with `bin_type: Log`;
  - NNN with `bin_type: LogSAS`.
- Not supported on the GPU: patches and jackknife or bootstrap covariances; three-catalogue NNN
  cross-correlations; other correlation types (NG, KK, GGG, …). Use `backend: cpu` for these;
  the same config file works.
