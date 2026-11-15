"""Legendre mode vs an independent NumPy brute force of sum_{triangles} P_l(cos phi)."""
import sys
import os
import time

import numpy as np
import treecorr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gpucorr  # noqa: E402
from gpucorr.engine_legendre import legendre_P  # noqa: E402

L, MINSEP, MAXSEP, NB = 1000., 50., 150., 50


def brute(p, per, lmax):
    binsize = np.log(MAXSEP / MINSEP) / NB
    M = np.zeros((lmax + 1, NB, NB))
    for i in range(len(p)):
        v = p - p[i]
        if per:
            v = v - per * np.round(v / per)
        d = np.sqrt((v * v).sum(1))
        ok = (d >= MINSEP) & (d < MAXSEP)
        ok[i] = False
        idx = np.nonzero(ok)[0]
        if len(idx) < 2:
            continue
        k = np.minimum(np.floor((np.log(d[idx]) - np.log(MINSEP)) / binsize).astype(int), NB - 1)
        u = v[idx] / d[idx, None]
        c = np.clip(u @ u.T, -1, 1)
        np.fill_diagonal(c, np.nan)
        jj, ll = np.nonzero(~np.isnan(c))
        P = legendre_P(lmax, c[jj, ll])[:lmax + 1]
        for l in range(lmax + 1):
            np.add.at(M[l], (k[jj], k[ll]), P[l])
    return 0.5 * M


def main(n=2000, lmax=12):
    rng = np.random.default_rng(3)
    p = rng.uniform(0, L, (n, 3))
    cat = treecorr.Catalog(x=p[:, 0], y=p[:, 1], z=p[:, 2])
    for metric in ('Euclidean', 'Periodic'):
        per = L if metric == 'Periodic' else None
        t = time.perf_counter()
        ref = brute(p, per, lmax)
        tb = time.perf_counter() - t
        kw = dict(min_sep=MINSEP, max_sep=MAXSEP, nbins=NB, nphi_bins=50, bin_type='LogSAS')
        if per:
            kw['period'] = per
        for prec in ('mixed', 'double'):
            g = gpucorr.GPUNNNCorrelation(gpu_opts=gpucorr.GPUOptions(gpu_method='legendre', gpu_lmax=lmax,
                                                                       gpu_precision=prec), **kw)
            g.process(cat, metric=metric)
            scale = np.abs(ref[0]).max()
            err = np.abs(g.legendre - ref).max() / scale
            print(f'{metric:9s} {prec:6s} lmax={lmax}: max |dM_l| / max M_0 = {err:.2e}   '
                  f'(monopole total {g.legendre[0].sum():.6e} vs {ref[0].sum():.6e}; brute {tb:.1f}s)')


if __name__ == '__main__':
    main()
