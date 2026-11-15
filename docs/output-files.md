# Output files

## TreeCorr-format files

`nn_file_name` and `nnn_file_name` are written by **TreeCorr's own `write`**, so the format is
identical for both backends. The first line is a `##` header with the run parameters, including
`tot`. The second is a `#` header with the column names.

### 2-point (`nn_file_name`)

| column | meaning |
|---|---|
| `r_nom` | nominal bin centre, exp of the log-bin centre |
| `meanr`, `meanlogr` | weighted mean of r and of ln r of the pairs in the bin |
| `xi`, `sigma_xi` | Landy–Szalay ξ and its shot-noise error |
| `DD`, `RR`, `DR` | pair weights of each term (RR, DR scaled by TreeCorr's tot ratios) |
| `npairs` | number of data pairs |

### 3-point (`nnn_file_name`, `bin_type: LogSAS`)

One row per (d₂, d₃, φ) bin, nbins × nbins × nphi_bins rows:

| column | meaning |
|---|---|
| `d2_nom`, `d3_nom`, `phi_nom` | nominal bin centres |
| `meand1`, `meanlogd1` | mean of d₁ = \|P₂P₃\| and of ln d₁ |
| `meand2`, `meanlogd2`, `meand3`, `meanlogd3` | the same for d₂ = \|P₁P₃\| and d₃ = \|P₁P₂\| |
| `meanphi` | mean opening angle at P₁ |
| `zeta`, `sigma_zeta` | full estimator (DDD − RDD + DRR − RRR)/RRR and its shot-noise error |
| `DDD`, `RRR`, `DRR`, `RDD` | triangle weights of each term (scaled by tot ratios) |
| `ntri` | number of data triangles |

Each triangle is counted once per vertex as P₁, as in TreeCorr.

With `gpu_method: legendre` the columns are the same, but:
- the counts come from the multipole reconstruction and are symmetric in d₂ ↔ d₃;
- `meanphi` and `meand1` are bin-centre values.

See [Methods](methods.md#the-legendre-method-slepian--eisenstein-2015).

Reading back in Python:

```python
import numpy as np
z = np.genfromtxt('3pcf_periodic.txt', names=True, skip_header=1)
zeta = z['zeta'].reshape(50, 50, 50)      # (d2 bin, d3 bin, phi bin)
```

The file can also be read with `treecorr.NNNCorrelation(...).read(...)`.

## The Legendre file

With `gpu_method: legendre`, a second file `<nnn_file_name stem>_legendre<ext>` is written,
e.g. `3pcf_periodic_legendre.txt`. It has one row per (d₂ bin, d₃ bin, ℓ):

| column | meaning |
|---|---|
| `d2_nom`, `d3_nom` | nominal radial bin centres |
| `l` | multipole order, 0 … `gpu_lmax` |
| `DDD_l` | Σ over data triangles of w · P_ℓ(cos φ), counting each triangle once per vertex as P₁, symmetrised and halved over d₂ ↔ d₃ |
| `RRR_l`, `DRR_l`, `RDD_l` | the same for the random and cross terms (unscaled) |
| `Z_l` | DDD_l − RDD_l·f_RDD + DRR_l·f_DRR − RRR_l·f_RRR, with TreeCorr's tot ratios f |
| `zeta_l` | Z_l / (RRR_0 · f_RRR) |

Conventions:

- `DDD_0` summed over all bins equals the total weight of the φ-binned output, i.e. Σ `DDD`
  before scaling.
- The angular dependence is recovered as ζ(φ) ≈ Σ_ℓ (2ℓ+1) ζ_ℓ P_ℓ(cos φ).
  - The Slepian–Eisenstein convention ζ(φ) = Σ_ℓ ζ_ℓ^SE P_ℓ(cos φ) corresponds to
    ζ_ℓ^SE = (2ℓ+1) ζ_ℓ.
- The RRR monopole normalisation is exact for an isotropic RRR, such as a periodic box. For a
  survey with a mask, use the columns `DDD_l` … `RDD_l` with your own edge correction (e.g. the
  Slepian–Eisenstein matrix inversion).

Reading in Python:

```python
import numpy as np
L = np.loadtxt('3pcf_periodic_legendre.txt')
nb, lmax = 50, int(L[:, 2].max())
zeta_l = L[:, -1].reshape(nb, nb, lmax + 1).transpose(2, 0, 1)   # (l, d2 bin, d3 bin)
```

## Objects in memory

From the Python API, the same quantities are arrays on the correlation objects: `dd.xi`,
`dd.npairs`, `ddd.weight`, `ddd.ntri`, `ddd.meand2`, …, and with Legendre `ddd.legendre`.
See [Python API](python-api.md).
