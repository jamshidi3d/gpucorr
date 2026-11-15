# Python API

```python
import gpucorr
gpucorr.GPUNNCorrelation      # treecorr.NNCorrelation  subclass, pair counting on the GPU
gpucorr.GPUNNNCorrelation     # treecorr.NNNCorrelation subclass, triangle counting on the GPU
gpucorr.GPUOptions            # GPU settings (method, precision, memory, device)
gpucorr.corr2(config)         # like treecorr.corr2, honouring `backend`
gpucorr.corr3(config)         # like treecorr.corr3, honouring `backend`
gpucorr.run_config(config)    # corr2 and/or corr3, depending on nn_file_name / nnn_file_name
```

## `GPUOptions`

```python
opts = gpucorr.GPUOptions(
    gpu_method='exact',       # 'exact' | 'legendre'
    gpu_lmax=10,              # legendre only, 0..63
    gpu_precision='mixed',    # 'mixed' | 'double'
    max_vram_gb=None,         # e.g. 12; None = half of the free physical VRAM
    gpu_device=0,
    gpu_guard=1e-5,
    gpu_block=256,
)
opts = gpucorr.GPUOptions.from_config(config_dict)   # same keys as in the config file
```

## Correlation classes

`GPUNNCorrelation` and `GPUNNNCorrelation` take TreeCorr's usual arguments: a config dict and/or
keyword arguments such as `min_sep`, `max_sep`, `nbins`, `nphi_bins`, `bin_type` and `period`.
They also take one extra keyword, `gpu_opts`. Without it, the options are read from the
`gpu_*` keys of the config dict.

They **are** TreeCorr objects. `process`, `finalize`, `calculateXi`, `calculateZeta`, `write`,
`read`, the `weight` / `ntri` / `meand2` / … arrays and the variance attributes all behave as in
TreeCorr. Only the accumulation inside `process_auto` / `process_cross` /
`process_cross12` runs on the GPU.

### 2-point function

```python
import treecorr, gpucorr

data = treecorr.Catalog('input/data.data', x_col=1, y_col=2, z_col=3, w_col=4)
rand = treecorr.Catalog('input/random.txt', x_col=1, y_col=2, z_col=3)
kw = dict(min_sep=50, max_sep=150, nbins=50, period=1000)

dd = gpucorr.GPUNNCorrelation(**kw); dd.process(data, metric='Periodic')
dr = gpucorr.GPUNNCorrelation(**kw); dr.process(data, rand, metric='Periodic')
rr = gpucorr.GPUNNCorrelation(**kw); rr.process(rand, metric='Periodic')

xi, varxi = dd.calculateXi(rr=rr, dr=dr)
dd.write('xi.txt', rr=rr, dr=dr)
```

### 3-point function, exact

```python
kw3 = dict(min_sep=50, max_sep=150, nbins=50, nphi_bins=50, bin_type='LogSAS', period=1000)
opts = gpucorr.GPUOptions(gpu_method='exact', max_vram_gb=12)

def nnn():
    return gpucorr.GPUNNNCorrelation(gpu_opts=opts, **kw3)

ddd = nnn(); ddd.process(data, metric='Periodic')
rrr = nnn(); rrr.process(rand, metric='Periodic')
drr = nnn(); drr.process(data, rand, ordered=False, metric='Periodic')
rdd = nnn(); rdd.process(rand, data, ordered=False, metric='Periodic')

zeta, varzeta = ddd.calculateZeta(rrr=rrr, drr=drr, rdd=rdd)   # shape (nbins, nbins, nphi_bins)
ddd.write('zeta.txt', rrr=rrr, drr=drr, rdd=rdd)
```

`ordered=False` for DRR / RDD matches what `corr3` does: every vertex can be the data point.

### 3-point function, Legendre multipoles

```python
opts = gpucorr.GPUOptions(gpu_method='legendre', gpu_lmax=10)
# ... same four process calls as above ...

ddd.legendre          # ndarray (lmax+1, nbins, nbins): sum over triangles of w * P_l(cos phi)
from gpucorr.correlation import legendre_zeta
zeta_l, Z_l = legendre_zeta(ddd, rrr=rrr, drr=drr, rdd=rdd)   # zeta_l: (lmax+1, nbins, nbins)

ddd.write('zeta.txt', rrr=rrr, drr=drr, rdd=rdd)   # also writes zeta_legendre.txt
```

`ddd.weight`, `ddd.ntri` and the other arrays hold the φ-binned reconstruction, so
`calculateZeta` and `write` work unchanged. The meaning of every quantity is in
[Output files](output-files.md#the-legendre-file).

### Reusing random terms across many mocks

RRR depends only on the randoms. Compute it once and reuse it for every data catalogue:

```python
rrr = nnn(); rrr.process(rand, metric='Periodic')        # once
for fname in mock_files:
    data = treecorr.Catalog(fname, x_col=1, y_col=2, z_col=3)
    ddd = nnn(); ddd.process(data, metric='Periodic')
    drr = nnn(); drr.process(data, rand, ordered=False, metric='Periodic')
    rdd = nnn(); rdd.process(rand, data, ordered=False, metric='Periodic')
    ddd.write(fname + '.zeta', rrr=rrr, drr=drr, rdd=rdd)
```

## Diagnostics: `gpu_stats`

After `process`, each object carries `obj.gpu_stats`:

| attribute | meaning |
|---|---|
| `time` | GPU wall time of this object's processing (s) |
| `batches` | number of memory batches used |
| `peak_bytes` | peak GPU memory held by CuPy's pool |
| `ntrip` | neighbour pairs processed (3pt) |
| `nexact` | triangles re-evaluated in fp64 near bin edges (exact, mixed mode) |
| `t_nbr`, `t_tri` | time spent building neighbour lists / counting |

## Config-driven use from Python

```python
from treecorr.config import read_config
import gpucorr

cfg = read_config('configs/p3pcfconfig.yaml')
cfg.update(backend='cuda', gpu_method='legendre', gpu_lmax=10)
gpucorr.run_config(cfg)        # runs exactly what run_corr.py would
```
