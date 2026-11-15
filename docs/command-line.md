# Command line

There are three equivalent entry points:

```bash
python run_corr.py <config> [key=value ...] [-v N] [-l LOGFILE]
gpucorr <config> [key=value ...]              # installed by `pip install -e .`
python -m gpucorr <config> [key=value ...]
```

| argument | meaning |
|---|---|
| `<config>` | a TreeCorr config file (`.yaml`, `.json` or params) |
| `key=value` | overrides or adds a config key, exactly like TreeCorr's `corr2` / `corr3` |
| `-v N`, `--verbosity N` | verbosity 0–3 (same as `verbose=N`) |
| `-l FILE`, `--log_file FILE` | write the log to `FILE` |

The program does the following:
- runs the 2PCF if the config contains `nn_file_name`, and the 3PCF if it contains `nnn_file_name`;
- dispatches each to TreeCorr (`backend: cpu`) or to gpucorr (`backend: cuda`);
- prints the total wall time at the end.

## Examples

```bash
# 2PCF on the GPU, writing to a different file
python run_corr.py configs/p2pcfconfig.yaml backend=cuda nn_file_name=xi_gpu.txt

# 3PCF, Legendre multipoles up to l = 24, capped at 12 GB of GPU memory
python run_corr.py configs/p3pcfconfig.yaml backend=cuda gpu_method=legendre gpu_lmax=24 max_vram_gb=12

# 3PCF exact, all-fp64 geometry
python run_corr.py configs/p3pcfconfig.yaml backend=cuda gpu_precision=double

# other catalogues, Euclidean metric (period must then be removed or unset)
python run_corr.py configs/p3pcfconfig.yaml backend=cuda file_name=mock.txt rand_file_name=rand.txt metric=Euclidean

# exact CPU reference with TreeCorr (slow; use a subsample)
python run_corr.py configs/p3pcfconfig.yaml backend=cpu bin_slop=0 algo=triangle nnn_file_name=zeta_cpu.txt
```

Overriding a key that is present in the file replaces it. Keys that are absent are added.
Values are parsed as YAML scalars (`12` is a number, `true` a boolean, `cuda` a string).

## Batch runs

A shell loop over many mocks can reuse one config file:

```bash
for i in $(seq -w 1 100); do
  python run_corr.py configs/p3pcfconfig.yaml backend=cuda gpu_method=legendre \
         file_name=mocks/mock_$i.txt nnn_file_name=out/zeta_$i.txt
done
```

Each call compiles nothing new after the first one (CuPy caches the kernels on disk). Loading
a catalogue and building its cell list takes well under a second.

For many catalogues **in one Python process**, it is more efficient to use the
[Python API](python-api.md) and reuse the random-catalogue terms (RRR), which don't change
between mocks.
