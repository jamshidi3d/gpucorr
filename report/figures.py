#!/usr/bin/env python
"""Data loading, extrapolation and figure helpers for the report (used by make_report_tex.py)."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# palette (reference categorical order; text in neutral inks)
C_GPU, C_DEF, C_TRI, C_EXA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
TERMS = ('DD', 'DR', 'RR', 'DDD', 'DRR', 'RDD', 'RRR')
MODES = [('default', 'TreeCorr default (NNN multipole)', C_DEF),
         ('triangle', 'TreeCorr triangle, default bin_slop', C_TRI),
         ('exact', 'TreeCorr exact, bin_slop=0', C_EXA)]

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
                     'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False})


def load():
    with open(os.path.join(ROOT, 'tests', 'validation_report.json')) as f:
        val = json.load(f)
    with open(os.path.join(ROOT, 'bench', 'results.json')) as f:
        bench = json.load(f)
    return val, bench


def fit(ns, ts):
    ns, ts = np.asarray(ns, float), np.asarray(ts, float)
    m = np.isfinite(ts) & (ts > 0.05)
    if m.sum() < 2:
        return None
    return np.polyfit(np.log(ns[m]), np.log(ts[m]), 1)


def cpu_extrap(rows, mode, n):
    """Per-term power-law extrapolation of a CPU mode (each term fitted on its own measured
    points, then summed).  Fitting the total instead would be dominated by the N_rand-independent
    DDD term and underestimate the large-N cost."""
    key = 'cpu_' + mode
    tot = 0.
    for k in TERMS:
        pts = [(r['nrand'], r[key][k]) for r in rows if r.get(key)]
        if not pts:
            return None
        if len(pts) == 1 or k in ('DD', 'DDD'):
            alpha = 0. if k in ('DD', 'DDD') else 1.
            n0, t0 = pts[-1]
        else:
            ns_, ts_ = np.log([p[0] for p in pts]), np.log([max(p[1], 1e-3) for p in pts])
            alpha = float(np.polyfit(ns_, ts_, 1)[0])
            n0, t0 = pts[-1]
        tot += t0 * (n / n0) ** alpha
    return tot


def fig_scaling(rows, path):
    ns = np.array([r['nrand'] for r in rows])
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    g = np.array([r['gpu_total'] for r in rows])
    ax.plot(ns, g, '-o', color=C_GPU, lw=2, ms=6, label='GPU (gpucorr, exact)')
    for mode, lab, col in MODES:
        key = 'cpu_%s_total' % mode
        t = np.array([r.get(key) if r.get(key) is not None else np.nan for r in rows], float)
        ok = np.isfinite(t)
        if ok.sum() == 0:
            continue
        ax.plot(ns[ok], t[ok], '-o', color=col, lw=2, ms=6, label=lab + ' (measured)')
        if (~ok).any():
            nn = np.array([ns[ok][-1]] + list(ns[~ok & (ns > ns[ok][-1])]))
            yy = np.array([t[ok][-1]] + [cpu_extrap(rows, mode, x) for x in nn[1:]])
            ax.plot(nn, yy, '--', color=col, lw=1.5, label=lab + ' (extrapolated)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('number of random points N_rand (N_data = %d)' % rows[0]['ndata'])
    ax.set_ylabel('wall time, all 7 terms (s)')
    ax.set_xlim(ns[0] / 1.3, ns[-1] * 1.3)
    ax.legend(fontsize=7, loc='upper left')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_terms(rows, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    ns = [r['nrand'] for r in rows]
    x = np.arange(len(ns))
    cols = ['#9fc3ee', '#5f9fe3', C_GPU, '#c9c8c2', '#8e8d88', '#52514e', '#0b0b0b']
    bottom = np.zeros(len(ns))
    totals = np.array([r['gpu_total'] for r in rows])
    for k, c in zip(TERMS, cols):
        v = 100 * np.array([r['gpu'][k] for r in rows]) / totals
        ax.bar(x, v, 0.6, bottom=bottom, color=c, label=k, edgecolor='white', linewidth=1)
        bottom += v
    for xi, t in zip(x, totals):
        ax.annotate(f'{t:.1f} s' if t < 100 else f'{t/60:.1f} min', (xi, 100), xytext=(0, 3),
                    textcoords='offset points', ha='center', fontsize=7, color=INK)
    ax.set_xticks(x, [f'{n:,}' for n in ns])
    ax.set_xlabel('N_rand')
    ax.set_ylabel('share of GPU time (%)')
    ax.set_ylim(0, 112)
    ax.legend(ncol=7, fontsize=7, loc='upper left', bbox_to_anchor=(0, 1.15))
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_multipole(path, n=20000):
    """phi distribution of RRR (summed over d2, d3) for the three algorithms, 3D periodic box."""
    import treecorr
    import gpucorr
    rng = np.random.default_rng(11)
    r = np.load(os.path.join(ROOT, 'bench', '_random_cache.npy')) if os.path.exists(
        os.path.join(ROOT, 'bench', '_random_cache.npy')) else np.loadtxt(os.path.join(ROOT, 'input', 'random.txt'))
    r = r[rng.choice(len(r), n, replace=False)]
    cat = treecorr.Catalog(x=r[:, 0], y=r[:, 1], z=r[:, 2])
    kw = dict(min_sep=50., max_sep=150., nbins=50, nphi_bins=50, bin_type='LogSAS', period=1000.)
    ex = gpucorr.GPUNNNCorrelation(**kw)
    ex.process(cat, metric='Periodic')
    tr = treecorr.NNNCorrelation(**kw)
    tr.process(cat, metric='Periodic', algo='triangle')
    mp = treecorr.NNNCorrelation(**kw)
    mp.process(cat, metric='Periodic')
    phi = np.degrees(ex.phi[0, 0])
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(phi, ex.weight.sum((0, 1)), '-', color=C_GPU, lw=2, label='GPU exact')
    ax.plot(phi, tr.weight.sum((0, 1)), '--', color=C_TRI, lw=2, label='TreeCorr triangle')
    ax.plot(phi, mp.weight.sum((0, 1)), '-', color=C_DEF, lw=2, label='TreeCorr multipole (default)')
    ax.set_xlabel('opening angle φ at P1 (deg)')
    ax.set_ylabel('RRR triangles per φ bin')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='lower center')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    m = ex.weight > 300
    return dict(n=n, mp_med=float(np.median(np.abs(mp.weight[m] / ex.weight[m] - 1))),
                tr_med=float(np.median(np.abs(tr.weight[m] / ex.weight[m] - 1))),
                mp_tot=float(mp.weight.sum() / ex.weight.sum() - 1),
                tr_tot=float(tr.weight.sum() / ex.weight.sum() - 1))


def fig_leg_timing(tim, path):
    ns = np.array([r['nrand'] for r in tim])
    terms = ('DDD', 'DRR', 'RDD', 'RRR')
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    ex = np.array([sum(r['exact'][k] for k in terms) if r.get('exact') else np.nan for r in tim])
    ax.plot(ns, ex, '-o', color=C_GPU, lw=2, ms=6, label='GPU exact (phi-binned counts)')
    for key, lab, col in (('leg10', 'GPU Legendre, lmax = 10', C_DEF), ('leg49', 'GPU Legendre, lmax = 49', C_TRI)):
        t = np.array([sum(r[key][k] for k in terms) if r.get(key) else np.nan for r in tim])
        ax.plot(ns, t, '-o', color=col, lw=2, ms=6, label=lab)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('N_rand (N_data = 10000)')
    ax.set_ylabel('3PCF wall time, DDD+DRR+RDD+RRR (s)')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
