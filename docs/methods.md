# Methods: exact and Legendre

| | `gpu_method: exact` | `gpu_method: legendre` |
|---|---|---|
| counterpart of | **TreeCorr** (`bin_slop=0`, `algo='triangle'`) | **Slepian & Eisenstein (2015)** |
| computes | triangle counts in (d₂, d₃, φ) bins | Legendre multipoles ζ_ℓ(d₂, d₃), ℓ ≤ ℓ_max |
| approximation | none | none for the multipoles; φ bins are reconstructed from a truncated series |
| cost | ∝ number of triangles ≈ N·n² | ∝ number of neighbour pairs × ℓ²_max ≈ N·n·ℓ²_max |
| best for | φ-binned ζ; bin-for-bin comparison with TreeCorr | multipole analyses; dense catalogues |

Here n is the mean number of neighbours within `max_sep`. The **2-point function is always
exact**: pairs counted in fp64 with TreeCorr's formulas.

## The exact method

A periodic (or, for the Euclidean metric, bounding-box) **cell list** finds the candidate
neighbours of every point. The cells are at least `max_sep/2` wide, and the search covers a
5×5×5 stencil.

Each candidate pair or triangle is then tested with **TreeCorr's leaf-level formulas**, taken
from TreeCorr 5.1's `Metric.h`, `BinType.h`, `Corr2.cpp` and `Corr3.cpp`:
- periodic displacement `d = p₂ − p₁`, then `while d > L/2: d −= L; while d < −L/2: d += L`;
- range test `min_sep ≤ d < max_sep` and log-bin index `⌊(ln d − ln min_sep)/bin_size⌋`;
- for triangles: d₁ = |P₂P₃|, d₂ = |P₁P₃|, d₃ = |P₁P₂|, cos φ = (d₂² + d₃² − d₁²)/(2 d₂ d₃),
  and φ < `max_phi`;
- each triangle is counted once per vertex as P₁, like TreeCorr's three rotations, with P₂/P₃
  labelled so that CCW(P₁, P₃, P₂) holds.

The grid only enumerates candidates and never decides whether a pair is counted. Nothing is
approximated, so the counts equal TreeCorr with `bin_slop=0`.

### Precision modes

**`mixed`** (default):
- Geometry runs in fp32 for speed.
- Any triangle whose φ lies within `gpu_guard` (1e-5 rad) of a bin edge, or that is nearly
  degenerate or ambiguously oriented, is **re-evaluated in fp64** with TreeCorr's formula. Bin
  assignment, and hence the counts, is therefore exact.
- Weights are exact for unit weights, and so is ζ, which depends only on the weights.
- The mean columns (`meand1`, `meanlogd1`, …, `meanphi`) carry fp32 rounding: about 1e-6
  relative.

**`double`**:
- Every triangle is evaluated in fp64 and summed in exact 64-bit fixed point.
- The means agree with TreeCorr to 1e-12–1e-15.
- It is about 10–20× slower on consumer GPUs (fp64 runs at 1/32–1/64 of the fp32 rate). On
  data-centre GPUs (A100, H100) the cost is much smaller.

### Periodic boxes and triangle orientation

In 3D, TreeCorr labels which side is d₂ and which is d₃ using
`sign[(r₁₃ × r₁₂) · p₁]`, a dot product with the **absolute position** of a vertex. In the
auto-correlation it evaluates this once, at whichever vertex its tree traversal happens to visit
first, and reuses it for the other two rotations.

For triangles that **cross the periodic boundary**, the sign can differ between vertices, so
TreeCorr's own labelling depends on its traversal order and is not reproducible point by point.
gpucorr always evaluates the rule at the triangle's own P₁. The consequences:

- **Euclidean metric:** gpucorr and TreeCorr agree bin for bin.
- **Periodic metric:**
  - Totals, all φ-marginals, and the counts summed over (d₂, d₃) + (d₃, d₂) are identical.
  - About 0.01–0.06 % of the triangle rotations sit in the mirror bin (d₃, d₂) instead of
    (d₂, d₃).
  - For a statistically isotropic box this is a pure relabelling.

## The Legendre method (Slepian & Eisenstein 2015)

For every centre c and radial bin k, gpucorr accumulates the spherical-harmonic coefficients

    a_ℓm(c, k) = Σ_{j ∈ k} w_j Y_ℓm(r̂_cj)          (m ≥ 0, normalised Y_ℓm, stable recursion)

By the addition theorem, the sum of P_ℓ(cos φ) over all pairs of neighbours in bins (k₂, k₃) is

    M_ℓ(k₂,k₃) = Σ_c w_c · 4π/(2ℓ+1) · [a_ℓ0 a*_ℓ0 + 2 Re Σ_{m>0} a_ℓm(k₂) a*_ℓm(k₃)]  − self-pairs

The sum over all centres is a single complex matrix product per ℓ (cuBLAS). The same task
decomposition as the exact method is used: P₁ = centre, every triangle once per vertex. The
monopole M₀ is therefore exactly the exact method's triangle total.

From the multipoles gpucorr produces:
1. **Multipoles of every term**, DDD_ℓ, RRR_ℓ, DRR_ℓ and RDD_ℓ, exact up to round-off, plus
   ζ_ℓ = Z_ℓ / (RRR₀ f) with Z_ℓ = DDD_ℓ − RDD_ℓ f + DRR_ℓ f − RRR_ℓ f (TreeCorr's tot factors f).
   Then ζ(φ) ≈ Σ_ℓ (2ℓ+1) ζ_ℓ P_ℓ(cos φ). Normalising by the RRR monopole is exact when RRR is
   isotropic in φ (periodic box); for masked surveys use the full RRR_ℓ.
2. **A φ-binned reconstruction** placed in TreeCorr's arrays, so the usual output file and
   `calculateZeta` work. Each bin gets the truncated series integrated over its range in
   μ = cos φ. These bins are symmetric in d₂ ↔ d₃, because the method has no 3D orientation.
   - `meand2`, `meanlogd2`, `meand3` and `meanlogd3` are exact per (d₂, d₃) bin.
   - `meanphi` is the bin centre, and `meand1` follows from the law of cosines.

### Choosing `gpu_lmax`

| `gpu_lmax` | use |
|---|---|
| 5–10 | standard 3PCF multipole analyses; fastest |
| 20–30 | finer angular structure |
| ≈ `nphi_bins` | φ-binned output close to exact counting (but exact counting may then be faster) |

The truncated series smooths the shot noise of the counts at high ℓ. In φ-bins, the
reconstruction therefore differs from exact counting by roughly the per-bin noise. That is
filtering, not bias. In multipole space the two methods agree, up to the binning of the exact
method; see the report, Sec. 5. The φ-binned reconstruction can give small negative values in
sparse bins (Gibbs-like behaviour); the multipoles themselves are exact.

## A warning about TreeCorr's default 3PCF algorithm

For `bin_type: LogSAS`, TreeCorr's default `algo` is `multipole`. For **3D (x, y, z)
catalogues**, TreeCorr 5.1.4's multipole path does not reproduce the opening-angle
distribution:
- uniform randoms come out with a flat φ distribution instead of sin φ;
- the median per-bin error is 32 %;
- raising `max_n` does not change it, so this is not truncation;
- in 2D the same algorithm agrees with the `triangle` algorithm to about 1 %.

When you run the 3PCF of a 3D catalogue on the CPU, set `algo: triangle`. gpucorr's two methods
are not affected.
