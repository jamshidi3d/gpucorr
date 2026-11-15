#!/usr/bin/env python
"""Legendre (Slepian-Eisenstein) mode vs the exact GPU method: accuracy vs lmax and timing.

    python tests/compare_legendre.py [--nrand 50000] [--timing 10000,50000,100000,200000,500000]
Writes report/legendre_results.json and report/fig_leg_*.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import treecorr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import gpucorr  # noqa: E402
from gpucorr.engine_legendre import phi_bin_projector  # noqa: E402

KW = dict(min_sep=50., max_sep=150., nbins=50, nphi_bins=50, min_phi=0., max_phi=np.pi,
          bin_type='LogSAS', period=1000.)
LMAXES = [2, 4, 6, 8, 10, 16, 24, 32, 49]
C_EX, C_L1, C_L2, C_L3 = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'


def load(nd, nr, seed=7):
    data = np.loadtxt(os.path.join(ROOT, 'input', 'data.data'))[:nd]
    cache = os.path.join(ROOT, 'bench', '_random_cache.npy')
    rand = np.load(cache) if os.path.exists(cache) else np.loadtxt(os.path.join(ROOT, 'input', 'random.txt'))
    rand = rand[np.random.default_rng(seed).permutation(len(rand))[:nr]]
    dc = treecorr.Catalog(x=data[:, 0], y=data[:, 1], z=data[:, 2], w=data[:, 3])
    rc = treecorr.Catalog(x=rand[:, 0], y=rand[:, 1], z=rand[:, 2])
    return dc, rc


def run_terms(opts, dc, rc):
    out, tm = {}, {}
    for name, args, ordered in (('DDD', (dc,), True), ('DRR', (dc, rc), False),
                                ('RDD', (rc, dc), False), ('RRR', (rc,), True)):
        c = gpucorr.GPUNNNCorrelation(gpu_opts=opts, **KW)
        kw = dict(metric='Periodic')
        if len(args) == 2:
            kw['ordered'] = ordered
        t0 = time.perf_counter()
        c.process(*args, **kw)
        tm[name] = time.perf_counter() - t0
        out[name] = c
    return out, tm


def zeta_from(w, tots):
    rrrf = tots['DDD'] / tots['RRR']
    drrf = tots['DDD'] / tots['DRR']
    rddf = tots['DDD'] / tots['RDD']
    den = w['RRR'] * rrrf
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (w['DDD'] - w['RDD'] * rddf + w['DRR'] * drrf - w['RRR'] * rrrf) / den
    return z


def sym(a):
    return 0.5 * (a + a.transpose(1, 0, 2))


def accuracy(nd, nr):
    dc, rc = load(nd, nr)
    ex, tex = run_terms(gpucorr.GPUOptions(), dc, rc)
    lg, tlg = run_terms(gpucorr.GPUOptions(gpu_method='legendre', gpu_lmax=max(LMAXES)), dc, rc)
    tots = {k: ex[k].tot for k in ex}
    wex = {k: sym(ex[k].weight) for k in ex}
    zex = zeta_from(wex, tots)
    # statistical error of zeta (TreeCorr shot-noise estimate on the exact run)
    zt, varz = ex['DDD'].calculateZeta(rrr=ex['RRR'], drr=ex['DRR'], rdd=ex['RDD'])
    sig = np.sqrt(sym(varz) / 2)        # symmetrising averages two bins
    good = wex['RRR'] > 100
    ro = ex['DDD']._ro
    res = dict(ndata=nd, nrand=nr, lmax=[], rrr_med_rel=[], ddd_med_rel=[], rrr_phimarg_max_rel=[],
               zeta_rms_abs=[], zeta_rms_over_sigma=[], zeta_max_over_sigma=[],
               mono_rel={k: float(np.max(np.abs(lg[k].legendre[0].sum() / wex[k].sum() - 1))) for k in ex},
               t_exact=tex, t_leg49=tlg)
    zrec_store = {}
    for L in LMAXES:
        J = phi_bin_projector(L, ro.min_u, ro.max_u, ro.nubins)
        wl = {k: np.einsum('lab,lk->abk', lg[k].legendre[:L + 1], J) for k in lg}
        zl = zeta_from(wl, tots)
        zrec_store[L] = zl
        d = (zl - zex)[good]
        res['lmax'].append(L)
        res['rrr_med_rel'].append(float(np.median(np.abs(wl['RRR'][good] / wex['RRR'][good] - 1))))
        gd = wex['DDD'] > 100
        res['ddd_med_rel'].append(float(np.median(np.abs(wl['DDD'][gd] / wex['DDD'][gd] - 1))) if gd.any() else np.nan)
        res['rrr_phimarg_max_rel'].append(float(np.max(np.abs(wl['RRR'].sum((0, 1)) / wex['RRR'].sum((0, 1)) - 1))))
        res['zeta_rms_abs'].append(float(np.sqrt(np.mean(d ** 2))))
        res['zeta_rms_over_sigma'].append(float(np.sqrt(np.mean((d / sig[good]) ** 2))))
        res['zeta_max_over_sigma'].append(float(np.max(np.abs(d / sig[good]))))
    # multipole space: zeta_l from the Legendre mode vs zeta_l projected from the exact binned
    # counts (bin-averaged P_l in mu; differs only by the phi binning of the exact method)
    Lc = 10
    J = phi_bin_projector(Lc, ro.min_u, ro.max_u, ro.nubins)
    Pbar = J / ((2 * np.arange(Lc + 1) + 1)[:, None] * J[0][None, :])
    Xex = {k: np.einsum('abk,lk->lab', wex[k], Pbar) for k in wex}
    Xlg = {k: lg[k].legendre[:Lc + 1] for k in lg}

    def zl(X):
        rrrf, drrf, rddf = (tots['DDD'] / tots[k] for k in ('RRR', 'DRR', 'RDD'))
        Z = X['DDD'] - X['RDD'] * rddf + X['DRR'] * drrf - X['RRR'] * rrrf
        return Z / (X['RRR'][0] * rrrf)[None]
    zle, zll = zl(Xex), zl(Xlg)
    g2 = wex['RRR'].sum(2) > 1e4
    res['multipole_l'] = list(range(Lc + 1))
    res['zeta_l_rms'] = [float(np.sqrt(np.mean(zll[l][g2] ** 2))) for l in range(Lc + 1)]
    res['zeta_l_rel_diff'] = [float(np.sqrt(np.mean((zll[l] - zle[l])[g2] ** 2)) /
                                    np.sqrt(np.mean(zll[l][g2] ** 2))) for l in range(Lc + 1)]
    res['zeta_rms'] = float(np.sqrt(np.mean(zex[good] ** 2)))
    res['sigma_median'] = float(np.median(sig[good]))
    # figures: zeta(phi) in two (d2,d3) bins
    phi = np.degrees(ro.phi1d)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=False)
    for ax, (a, b) in zip(axs, [(25, 25), (10, 40)]):
        ax.errorbar(phi, zex[a, b], yerr=sig[a, b], fmt='o', ms=3, color=C_EX, ecolor='#9fc3ee',
                    label='exact (±1σ shot noise)')
        for L, c in ((10, C_L1), (24, C_L2), (49, C_L3)):
            ax.plot(phi, zrec_store[L][a, b], '-', color=c, lw=1.6, label=f'Legendre, lmax={L}')
        rn = ro.min_sep * np.exp((np.array([a, b]) + 0.5) * ro.bin_size)
        ax.set_title(f'd2 ≈ {rn[0]:.0f}, d3 ≈ {rn[1]:.0f}', fontsize=9)
        ax.set_xlabel('φ (deg)')
    axs[0].set_ylabel('zeta')
    axs[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'report', 'fig_leg_zeta.png'), dpi=200)
    fig.savefig(os.path.join(ROOT, 'report', 'fig_leg_zeta.pdf'))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    ax.plot(res['lmax'], res['zeta_rms_over_sigma'], '-o', color=C_EX, lw=2, label='rms |Δzeta| / σ_zeta')
    ax.plot(res['lmax'], res['rrr_med_rel'], '-o', color=C_L1, lw=2, label='RRR: median per-bin rel. error')
    ax.set_yscale('log')
    ax.set_xlabel('lmax')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, 'report', 'fig_leg_conv.png'), dpi=200)
    fig.savefig(os.path.join(ROOT, 'report', 'fig_leg_conv.pdf'))
    plt.close(fig)
    return res


def timing(nd, nrs, lmaxes=(10, 49)):
    bench = json.load(open(os.path.join(ROOT, 'bench', 'results.json')))
    exact = {r['nrand']: r['gpu'] for r in bench['rows']}
    out = []
    for nr in nrs:
        dc, rc = load(nd, nr)
        row = dict(nrand=nr, exact=exact.get(nr))
        for L in lmaxes:
            o = gpucorr.GPUOptions(gpu_method='legendre', gpu_lmax=L)
            _, tm = run_terms(o, dc, rc)
            row['leg%d' % L] = tm
            print(nr, L, {k: round(v, 2) for k, v in tm.items()}, flush=True)
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ndata', type=int, default=10000)
    ap.add_argument('--nrand', type=int, default=50000)
    ap.add_argument('--timing', default='10000,20000,50000,100000,200000,500000')
    ap.add_argument('--timing-only', action='store_true')
    a = ap.parse_args()
    path = os.path.join(ROOT, 'report', 'legendre_results.json')
    if a.timing_only and os.path.exists(path):
        acc = json.load(open(path))['accuracy']
    else:
        acc = accuracy(a.ndata, a.nrand)
    print(json.dumps({k: v for k, v in acc.items() if k not in ('t_exact', 't_leg49')}, indent=0), flush=True)
    tim = timing(a.ndata, [int(x) for x in a.timing.split(',')]) if a.timing else None
    if tim is None:      # keep a previous timing scan
        path = os.path.join(ROOT, 'report', 'legendre_results.json')
        tim = json.load(open(path)).get('timing', []) if os.path.exists(path) else []
    with open(os.path.join(ROOT, 'report', 'legendre_results.json'), 'w') as f:
        json.dump(dict(accuracy=acc, timing=tim), f, indent=1)


if __name__ == '__main__':
    main()
