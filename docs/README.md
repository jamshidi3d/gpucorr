# gpucorr documentation

gpucorr runs TreeCorr's count–count correlation functions on an NVIDIA GPU:
- the 2-point **NN** correlation, ξ(r);
- the 3-point **NNN** correlation with `bin_type: LogSAS`, ζ(d₂, d₃, φ).

It offers two 3-point methods:
- **`exact`**, the exact counterpart of TreeCorr;
- **`legendre`**, the multipole counterpart of the Slepian & Eisenstein (2015) estimator.

| Guide | Read it when you want to… |
|---|---|
| [Installation](installation.md) | install gpucorr locally or on Google Colab |
| [Quick start](quickstart.md) | run your first ξ and ζ in five minutes |
| [Configuration reference](configuration.md) | know every config key: TreeCorr keys and the `backend` / `gpu_*` keys |
| [Command line](command-line.md) | use `run_corr.py`, `gpucorr` or `python -m gpucorr`, with overrides |
| [Python API](python-api.md) | drive gpucorr from Python, or mix it with your own TreeCorr code |
| [Methods: exact and Legendre](methods.md) | understand what is computed, the precision modes and how the results relate to TreeCorr |
| [Output files](output-files.md) | read the TreeCorr-format output and the `*_legendre.txt` multipole file |
| [Memory and performance](memory-and-performance.md) | run on small or shared GPUs (12 GB Colab) and pick the fastest settings |
| [Validation and benchmarks](validation.md) | reproduce the consistency checks, the benchmark and the report |
| [Troubleshooting and FAQ](troubleshooting.md) | fix error messages, and see the limitations |

Measured speed and precision for all methods, compared with TreeCorr, are in the
[main README](../README.md#results) and in the full report,
[`report/gpucorr_vs_treecorr.pdf`](../report/gpucorr_vs_treecorr.pdf).
