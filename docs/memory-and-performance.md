# Memory and performance

## How memory is used

| object | size | lifetime |
|---|---|---|
| catalogue positions + weights + cell list | ≈ 40 B per point | whole run |
| per-centre neighbour lists (3pt) | 24 B per neighbour (mixed) / 4 B (double) | one batch |
| Legendre coefficients a_ℓm (legendre) | ≈ nbins × (ℓ_max+1)(ℓ_max+2)/2 × 16 B per centre | one batch |
| result arrays | nbins² × nphi_bins × 9 × 8 B (≈ 9 MB for 50³) | whole run |

Centres are processed in **batches**, and each batch is sized to fit the memory budget:
- **default:** half of the free physical VRAM, after a reserve for the driver and display;
- **with `max_vram_gb: X`:** at most X GB.

The results do not depend on the budget; a smaller budget only means more batches. In the
benchmark, a 0.5 GB cap gave counts identical to the uncapped run, at the same speed.

### Google Colab (12–16 GB)

Set `max_vram_gb: 12`, or lower if other notebooks share the GPU. For catalogues of 10⁵–10⁶
points, memory is never the limiting factor.

### Windows laptops and shared GPUs

Under Windows (WDDM), the driver can silently page GPU memory out to system RAM when a program
asks for more than the card physically has. This is about 100× slower and shows up as "100 %
utilisation at very low power". gpucorr plans only within physical VRAM. If another program
holds a lot of GPU memory, set `max_vram_gb` explicitly.

Kernel launches are kept to about 0.25 s each, so a GPU that also drives the display does not
trip the Windows driver watchdog (TDR).

## How the cost scales

At fixed box volume, with n ∝ N the mean number of neighbours:

| term | exact method | legendre |
|---|---|---|
| DD, DR, RR | ∝ N·n (pairs) | (2pt is always exact) |
| RRR, DRR | ∝ N·n² (triangles) | ∝ N·n·ℓ²_max |
| DDD, RDD | small when N_data ≪ N_rand | small |

Measured on an RTX 4070 Laptop GPU, with 10k data, 50 log bins from 50 to 150 and 50 φ bins in a
box of side 1000. These are 3PCF times only; the 2PCF adds seconds.

| N_rand | exact | legendre ℓ_max = 10 | legendre ℓ_max = 49 |
|---:|---:|---:|---:|
| 10,000 | 2.0 s | 0.9 s | 8.4 s |
| 100,000 | 32 s | 5.3 s | 63 s |
| 500,000 | 26 min | 66 s | 11 min |

The exact method sustains about 8 × 10⁹ neighbour pairs per second at high density.

## Choosing settings for speed

- **Need multipoles?** `gpu_method: legendre` with the smallest `gpu_lmax` your analysis uses.
  Its advantage grows with density.
- **Need φ-bins exactly?** Use `gpu_method: exact` with `gpu_precision: mixed`. `double` only
  pays off if you need the mean columns to 1e-12.
- **Bins and `max_sep`:**
  - Exact-method cost grows with the square of the number of neighbours within `max_sep`.
  - Halving `max_sep` cuts it about 64× (neighbours ∝ max_sep³).
  - The number of bins hardly matters.
- **Randoms:** the cost of RRR grows ∝ N_rand³ in a fixed box. Many analyses use
  N_rand ≈ 10–50 × N_data and reuse RRR across mocks (see
  [Python API](python-api.md#reusing-random-terms-across-many-mocks)).
- **Several GPUs:** run independent mocks on different devices with `gpu_device`.

## Expected speed on other GPUs

The kernels are fp32- and shared-memory-bound. The following are rough guides relative to the
RTX 4070 Laptop GPU, not measurements:

| GPU | expected speed |
|---|---|
| Colab T4 | ~0.5–0.7× |
| L4 | ~1.3–1.8× |
| A100 / H100 | ~2–4× (and much faster in `double`) |
