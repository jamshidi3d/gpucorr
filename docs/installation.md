# Installation

## Requirements

- An **NVIDIA GPU**, compute capability 6.0 (Pascal) or newer. That includes Colab's T4, L4 and
  A100 and any recent GeForce or RTX card.
- An **NVIDIA driver** with CUDA 12 or 13.
- **Python ≥ 3.9**, with numpy, TreeCorr ≥ 5.1, PyYAML and CuPy.

You do **not** need the CUDA toolkit or a C++ compiler. The kernels in `gpucorr/kernels.cu` are
compiled at runtime by NVRTC through CuPy. The first run compiles them (about 3–7 s); later runs
reuse CuPy's on-disk cache.

Which CUDA version does your driver support? Run `nvidia-smi` and read the `CUDA Version` field
in the header.

## Local install

```bash
git clone https://github.com/jamshidi3d/gpucorr
cd gpucorr
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -e ".[cuda12]"      # CUDA 12.x driver
# or
pip install -e ".[cuda13]"      # CUDA 13.x driver
```

The extras only choose the matching CuPy wheel (`cupy-cuda12x` or `cupy-cuda13x`). If CuPy is
already installed, `pip install -e .` is enough.

For the benchmark and report scripts, also install `pip install -e ".[dev]"`, which adds
matplotlib and pytest. Building the PDF report additionally needs a LaTeX distribution with
`latexmk`.

## Google Colab

1. Runtime → Change runtime type → **GPU**.
2. In a cell:
   ```python
   !git clone https://github.com/jamshidi3d/gpucorr
   %cd gpucorr
   !pip -q install -e .          # CuPy for CUDA 12 is preinstalled on Colab
   ```
3. Upload your catalogues into `input/`, or mount Google Drive and point `file_name` /
   `rand_file_name` at them.

[`notebooks/colab_gpucorr.ipynb`](../notebooks/colab_gpucorr.ipynb) contains these steps, a
12 GB-capped run, and the validation and benchmark scripts.

## Check the installation

```bash
python -c "import gpucorr, cupy; print(gpucorr.__version__, cupy.cuda.runtime.getDeviceCount(), 'GPU(s)')"
python tests/quick_check.py 1000 Periodic      # GPU vs TreeCorr exact on 1000 random points
```

`quick_check.py` should report `bins differing=0` for the 2-point function. For the periodic
3-point function it should report `sym bins differing=0`; see
[Methods → periodic orientation](methods.md#periodic-boxes-and-triangle-orientation).
