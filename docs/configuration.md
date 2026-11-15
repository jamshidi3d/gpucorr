# Configuration reference

gpucorr reads **TreeCorr's configuration files** (YAML, JSON or `key = value` params) and adds a
few keys of its own.
- With `backend: cpu`, the gpucorr keys are removed and the config goes to `treecorr.corr2` /
  `treecorr.corr3` unchanged.
- With `backend: cuda`, the same keys drive the GPU code.

Two example files are provided:
- [`configs/p2pcfconfig.yaml`](../configs/p2pcfconfig.yaml), the 2-point function (`nn_file_name`);
- [`configs/p3pcfconfig.yaml`](../configs/p3pcfconfig.yaml), the 3-point function (`nnn_file_name`).

Whether a run computes the 2PCF, the 3PCF or both depends on which of `nn_file_name` and
`nnn_file_name` is set.

## gpucorr keys

| key | values (default) | meaning |
|---|---|---|
| `backend` | `"cpu"` \| `"cuda"` (`"cpu"`) | `cpu` runs plain TreeCorr; `cuda` runs gpucorr (`"gpu"` is accepted as an alias). |
| `gpu_method` | `"exact"` \| `"legendre"` (`"exact"`) | 3PCF method; see [Methods](methods.md). Ignored for the 2PCF, which is always exact. |
| `gpu_lmax` | integer 0–63 (`10`) | highest multipole ℓ for `gpu_method: legendre`. |
| `gpu_precision` | `"mixed"` \| `"double"` (`"mixed"`) | `mixed`: fp32 geometry with an fp64 recheck near every bin edge (counts exact, means ≈1e-6). `double`: all fp64 (means ≈1e-12–1e-15, slower). |
| `max_vram_gb` | number (unset) | upper limit on the GPU memory gpucorr may plan for. Unset: half of the free physical VRAM. Results do not depend on it. |
| `gpu_device` | integer (`0`) | which GPU to use. |
| `gpu_guard` | radians (`1e-5`) | width of the band around φ-bin edges inside which `mixed` mode re-evaluates a triangle in fp64. Rarely changed. |
| `gpu_block` | integer (`256`) | CUDA block size of the pair kernel. Rarely changed. |

With `backend: cuda`, TreeCorr's tree-approximation keys `bin_slop`, `angle_slop`, `brute`,
`algo` and `max_n` are **ignored**, because counting is exact. A warning names any that are present.

## TreeCorr keys used by gpucorr

### Input catalogues

| key | example | note |
|---|---|---|
| `file_name` | `input/data.data` | data catalogue (a list of files is also accepted, as in TreeCorr) |
| `rand_file_name` | `input/random.txt` | random catalogue; enables DR/RR (2pt) and DRR/RDD/RRR (3pt) |
| `file_type` | `ASCII` | or `FITS` |
| `x_col`, `y_col`, `z_col` | `1`, `2`, `3` | 1-based column numbers (column names for FITS) |
| `w_col` | `4` | optional weights; unit weights if omitted |

Paths are relative to the directory you run from.

### Geometry

| key | example | note |
|---|---|---|
| `metric` | `"Periodic"` | `Periodic` or `Euclidean` (others are CPU-only) |
| `period` | `1000` | box side, or `xperiod` / `yperiod` / `zperiod` separately; required for `Periodic` |

Only 3D Cartesian catalogues (`x`, `y`, `z`) are supported on the GPU.

### Binning

| key | example | note |
|---|---|---|
| `min_sep`, `max_sep`, `nbins` | `50`, `150`, `50` | logarithmic bins in r (2pt) and in d₂, d₃ (3pt); `bin_size` may replace one of them |
| `bin_type` | `"Log"` (2pt) / `"LogSAS"` (3pt) | the only types supported on the GPU |
| `min_phi`, `max_phi`, `nphi_bins` | `0`, `3.14159…`, `50` | linear φ bins (3pt) |

### Output and logging

| key | example | note |
|---|---|---|
| `nn_file_name` | `2pcf_periodic.txt` | 2PCF output (computing it is triggered by this key) |
| `nnn_file_name` | `3pcf_periodic.txt` | 3PCF output; with `legendre` also `<name>_legendre.txt` |
| `nn_statistic`, `nnn_statistic` | `compensated` | TreeCorr's estimator choice (`compensated` adds DR / DRR+RDD) |
| `verbose` | `0`–`3` | logging level |
| `log_file` | path | write the log to a file |

All other TreeCorr keys are passed to TreeCorr as usual. Unknown keys raise TreeCorr's
"invalid parameter" error.

## Example: the bottom of `p3pcfconfig.yaml`

```yaml
backend: "cuda"
gpu_method: "legendre"
gpu_lmax: 10
gpu_precision: "mixed"
max_vram_gb: 12
```

Any key can also be overridden on the command line, e.g. `gpu_lmax=24`; see
[Command line](command-line.md).
