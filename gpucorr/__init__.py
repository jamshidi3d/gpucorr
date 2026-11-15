"""gpucorr -- exact GPU (CUDA) backend for TreeCorr NN and NNN (LogSAS) correlations.

    from gpucorr import GPUNNCorrelation, GPUNNNCorrelation, corr2, corr3, run_config
"""
from .correlation import GPUNNCorrelation, GPUNNNCorrelation
from .engine import GPUOptions
from .run import corr2, corr3, run_config, split_config, main

__version__ = '0.2.0'
