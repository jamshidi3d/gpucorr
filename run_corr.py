#!/usr/bin/env python
"""Run TreeCorr NN / NNN from a config file on the backend chosen by `backend: cpu|cuda`.

    python run_corr.py configs/p2pcfconfig.yaml
    python run_corr.py configs/p3pcfconfig.yaml backend=cuda nnn_file_name=3pcf_gpu.txt
"""
from gpucorr.run import main

if __name__ == '__main__':
    main()
