"""Backend dispatch: one config file, `backend: cpu` (TreeCorr) or `backend: cuda` (gpucorr).

    python run_corr.py configs/p3pcfconfig.yaml                 # uses `backend` from the file
    python run_corr.py configs/p3pcfconfig.yaml backend=cuda    # command-line override (corr2/corr3 style)
"""
import argparse
import contextlib
import logging
import sys
import time

import treecorr
import treecorr.exec_corr2 as _ec2
import treecorr.exec_corr3 as _ec3
from treecorr.config import read_config, parse_variable, setup_logger

from .correlation import GPU_KEYS, GPUNNCorrelation, GPUNNNCorrelation
from .engine import GPUOptions

BACKENDS = ('cpu', 'cuda')


def split_config(config):
    """Return (treecorr_config, backend, gpu_options_dict)."""
    cfg = dict(config)
    backend = str(cfg.pop('backend', 'cpu')).lower()
    if backend in ('gpu',):
        backend = 'cuda'
    if backend not in BACKENDS:
        raise ValueError("backend must be one of %s, got %r" % (BACKENDS, backend))
    gpu = {k: cfg.pop(k) for k in list(cfg) if k in GPU_KEYS}
    return cfg, backend, gpu


@contextlib.contextmanager
def _patched(module, name, value):
    old = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, old)


def _factory(cls, gpu_opts):
    def make(config=None, logger=None, **kwargs):
        return cls(config, logger=logger, gpu_opts=gpu_opts, **kwargs)
    return make


def _warn_ignored(cfg, logger, three_pt):
    ign = [k for k in ('bin_slop', 'angle_slop', 'algo', 'max_n', 'brute') if k in cfg]
    if ign:
        logger.warning("backend=cuda counts every %s exactly; ignoring %s",
                       'triangle' if three_pt else 'pair', ', '.join(ign))


def corr2(config, logger=None):
    """Like `treecorr.corr2`, honouring `backend`.  On cuda the NN part runs on the GPU."""
    cfg, backend, gpu = split_config(config)
    if logger is None:
        logger = setup_logger(cfg.get('verbose', 1), cfg.get('log_file', None))
    if backend == 'cpu':
        return treecorr.corr2(cfg, logger=logger)
    opts = GPUOptions.from_config(gpu)
    _warn_ignored(cfg, logger, False)
    cfg.pop('algo', None)
    cfg.pop('max_n', None)
    with _patched(_ec2, 'NNCorrelation', _factory(GPUNNCorrelation, opts)):
        return _ec2.corr2(cfg, logger=logger)


def corr3(config, logger=None):
    """Like `treecorr.corr3`, honouring `backend`.  On cuda the NNN part runs on the GPU."""
    cfg, backend, gpu = split_config(config)
    if logger is None:
        logger = setup_logger(cfg.get('verbose', 1), cfg.get('log_file', None))
    if backend == 'cpu':
        return treecorr.corr3(cfg, logger=logger)
    opts = GPUOptions.from_config(gpu)
    _warn_ignored(cfg, logger, True)
    cfg.pop('algo', None)
    cfg.pop('max_n', None)
    with _patched(_ec3, 'NNNCorrelation', _factory(GPUNNNCorrelation, opts)):
        return _ec3.corr3(cfg, logger=logger)


def run_config(config, logger=None):
    """Run whatever the config asks for: nn_file_name -> corr2, nnn_file_name -> corr3."""
    two = any(k in config for k in ('nn_file_name',))
    three = any(k in config for k in ('nnn_file_name',))
    if not (two or three):
        raise ValueError("config has neither nn_file_name nor nnn_file_name")
    if two:
        corr2(config, logger)
    if three:
        corr3(config, logger)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="TreeCorr NN / NNN with a selectable cpu (TreeCorr) or cuda (gpucorr) "
                    "backend.")
    p.add_argument('config_file')
    p.add_argument('variables', nargs='*', help='key=value overrides, e.g. backend=cuda')
    p.add_argument('-v', '--verbosity', type=int, default=None)
    p.add_argument('-l', '--log_file', default=None)
    a = p.parse_args(argv)
    config = read_config(a.config_file)
    for v in a.variables:
        parse_variable(config, v)
    if a.verbosity is not None:
        config['verbose'] = a.verbosity
    if a.log_file is not None:
        config['log_file'] = a.log_file
    t0 = time.perf_counter()
    run_config(config)
    print("[gpucorr] backend=%s  wall time %.2f s"
          % (str(config.get('backend', 'cpu')), time.perf_counter() - t0))


if __name__ == '__main__':
    main()
