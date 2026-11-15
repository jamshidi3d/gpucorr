#!/usr/bin/env python
"""Benchmark gpucorr (cuda backend) against TreeCorr (cpu backend) on the project catalogs.

For every N_rand (subsample of input/random.txt; data = first N_data rows of input/data.data):
  * GPU (exact):   DD, DR, RR  and  DDD, DRR, RDD, RRR  (as in corr2 / corr3)
  * CPU default:   TreeCorr with its default settings (bin_slop default; NNN multipole)
  * CPU exact:     TreeCorr bin_slop=0 (NNN algo='triangle')  -- the exact reference
CPU runs happen in a subprocess with a wall-clock cap; capped runs are reported as '>cap'
and extrapolated with the fitted power law (marked '~').

    python bench/benchmark.py --nrand 10000,20000,50000,100000,200000,500000 --cap 900
Writes bench/results.csv and bench/results.md.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

L = 1000.
KW2 = dict(min_sep=50., max_sep=150., nbins=50, period=L)
KW3 = dict(KW2, nphi_bins=50, min_phi=0., max_phi=np.pi, bin_type='LogSAS')
TERMS2 = ('DD', 'DR', 'RR')
TERMS3 = ('DDD', 'DRR', 'RDD', 'RRR')
CPU_MODES = ('default', 'triangle', 'exact')
CPU_LABEL = {'default': 'CPU TreeCorr default (NNN multipole)',
             'triangle': 'CPU TreeCorr triangle (default bin_slop)',
             'exact': 'CPU TreeCorr exact (bin_slop=0)'}


def load(nd, nr, seed=7):
    data = np.loadtxt(os.path.join(ROOT, 'input', 'data.data'))
    rand = np.load(_rand_cache())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(rand))
    return data[:nd], rand[perm[:nr]]


def _rand_cache():
    p = os.path.join(HERE, '_random_cache.npy')
    if not os.path.exists(p):
        np.save(p, np.loadtxt(os.path.join(ROOT, 'input', 'random.txt')))
    return p


def cats(d, r):
    import treecorr
    dc = treecorr.Catalog(x=d[:, 0], y=d[:, 1], z=d[:, 2], w=d[:, 3] if d.shape[1] > 3 else None)
    rc = treecorr.Catalog(x=r[:, 0], y=r[:, 1], z=r[:, 2])
    return dc, rc


def run_terms(make2, make3, dc, rc, algo=None, collect=None):
    """Run all 7 terms; return {term: seconds}."""
    t = {}
    objs = {}
    for name, args in (('DD', (dc,)), ('DR', (dc, rc)), ('RR', (rc,))):
        c = make2()
        t0 = time.perf_counter()
        c.process(*args, metric='Periodic')
        t[name] = time.perf_counter() - t0
        objs[name] = c
    for name, args, ordered in (('DDD', (dc,), True), ('DRR', (dc, rc), False),
                                ('RDD', (rc, dc), False), ('RRR', (rc,), True)):
        c = make3()
        kw = dict(metric='Periodic')
        if len(args) == 2:
            kw['ordered'] = ordered
        if algo:
            kw['algo'] = algo
        t0 = time.perf_counter()
        c.process(*args, **kw)
        t[name] = time.perf_counter() - t0
        objs[name] = c
    if collect is not None:
        collect.update(objs)
    return t


def cpu_worker(mode, nd, nr, out):
    import treecorr
    d, r = load(nd, nr)
    dc, rc = cats(d, r)
    if mode == 'default':        # TreeCorr defaults: bin_slop default, NNN algo = multipole
        t = run_terms(lambda: treecorr.NNCorrelation(**KW2), lambda: treecorr.NNNCorrelation(**KW3),
                      dc, rc)
    elif mode == 'triangle':     # bin_slop default, NNN algo = triangle
        t = run_terms(lambda: treecorr.NNCorrelation(**KW2), lambda: treecorr.NNNCorrelation(**KW3),
                      dc, rc, algo='triangle')
    else:                        # exact: bin_slop = 0, algo = triangle
        t = run_terms(lambda: treecorr.NNCorrelation(bin_slop=0, **KW2),
                      lambda: treecorr.NNNCorrelation(bin_slop=0, **KW3), dc, rc, algo='triangle')
    with open(out, 'w') as f:
        json.dump(t, f)


def cpu_run(mode, nd, nr, cap):
    out = os.path.join(HERE, f'_cpu_{mode}_{nd}_{nr}.json')
    if os.path.exists(out):
        os.remove(out)
    cmd = [sys.executable, os.path.abspath(__file__), '--worker', mode, '--ndata', str(nd),
           '--nrand-one', str(nr), '--out', out]
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, timeout=cap, check=True, capture_output=True)
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - t0
    with open(out) as f:
        t = json.load(f)
    os.remove(out)
    return t, time.perf_counter() - t0


def gpu_run(nd, nr, precision, max_vram_gb=None):
    import gpucorr
    d, r = load(nd, nr)
    dc, rc = cats(d, r)
    opts = gpucorr.GPUOptions(gpu_precision=precision, max_vram_gb=max_vram_gb)
    objs = {}
    t = run_terms(lambda: gpucorr.GPUNNCorrelation(gpu_opts=opts, **KW2),
                  lambda: gpucorr.GPUNNNCorrelation(gpu_opts=opts, **KW3), dc, rc, collect=objs)
    st = dict(peak_mib=max(o.gpu_stats.peak_bytes for o in objs.values()) / 2**20,
              batches=sum(o.gpu_stats.batches for o in objs.values()),
              triplets=sum(getattr(o.gpu_stats, 'ntrip', 0) for k, o in objs.items() if len(k) == 3),
              ntri_RRR=float(objs['RRR'].ntri.sum()))
    zeta, _ = objs['DDD'].calculateZeta(rrr=objs['RRR'], drr=objs['DRR'], rdd=objs['RDD'])
    xi, _ = objs['DD'].calculateXi(rr=objs['RR'], dr=objs['DR'])
    return t, st, objs, zeta, xi


def fit_power(ns, ts):
    ns, ts = np.asarray(ns, float), np.asarray(ts, float)
    m = np.isfinite(ts) & (ts > 0.05)
    if m.sum() < 2:
        return None
    a, b = np.polyfit(np.log(ns[m]), np.log(ts[m]), 1)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ndata', type=int, default=10000)
    ap.add_argument('--nrand', default='10000,20000,50000,100000,200000,500000')
    ap.add_argument('--cap', type=float, default=900., help='CPU wall-clock cap per N (s)')
    ap.add_argument('--cpu-exact-max', type=int, default=20000,
                    help='largest N_rand for which the (slow) CPU exact run is attempted')
    ap.add_argument('--skip-cpu', action='store_true')
    ap.add_argument('--skip-gpu', action='store_true', help='reuse bench/results_gpu.json')
    ap.add_argument('--resume', default=None, help='results_partial json whose rows are reused')
    ap.add_argument('--redo', default='', help='comma list of N_rand to recompute despite --resume')
    ap.add_argument('--precision', default='mixed')
    ap.add_argument('--vram-test', type=float, default=0.25,
                    help='repeat the smallest-but-one case with this max_vram_gb cap (0: skip)')
    ap.add_argument('--worker', default=None)
    ap.add_argument('--nrand-one', type=int, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    if a.worker:
        cpu_worker(a.worker, a.ndata, a.nrand_one, a.out)
        return

    import cupy as cp
    dev = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = dev['name'].decode() if isinstance(dev['name'], bytes) else dev['name']
    gpu_mem = dev['totalGlobalMem'] / 2**30
    import platform
    cpu_name = platform.processor()
    ncpu = os.cpu_count()

    nrs = [int(x) for x in a.nrand.split(',')]
    rows = []
    capped = set()
    done = {}
    if a.resume:
        with open(a.resume) as f:
            redo = {int(x) for x in a.redo.split(',') if x}
            for r in json.load(f):
                if r['nrand'] not in redo:
                    done[r['nrand']] = r
        for r in done.values():
            for m in CPU_MODES:
                if ('cpu_' + m) in r and r['cpu_' + m] is None:
                    capped.add(m)
    gpath = os.path.join(HERE, 'results_gpu.json')
    if a.skip_gpu:
        with open(gpath) as f:
            gold = {r['nrand']: r for r in json.load(f)}
    else:
        gpu_warm = gpu_run(1000, 2000, a.precision)   # compile + warm up
        del gpu_warm
        cp.get_default_memory_pool().free_all_blocks()
    for nr in nrs:
        if nr in done:
            rows.append(done[nr])
            continue
        print(f'== N_rand = {nr}', flush=True)
        if a.skip_gpu:
            row = dict(gold[nr])
        else:
            tg, st, objs, zeta, xi = gpu_run(a.ndata, nr, a.precision)
            row = dict(nrand=nr, ndata=a.ndata, gpu=tg, gpu_total=sum(tg.values()), **st)
            print('  GPU', {k: round(v, 3) for k, v in tg.items()}, st, flush=True)
            del objs
            cp.get_default_memory_pool().free_all_blocks()
            if a.skip_cpu:
                old = []
                if os.path.exists(gpath):
                    with open(gpath) as f:
                        old = [r for r in json.load(f) if r['nrand'] != nr]
                with open(gpath, 'w') as f:
                    json.dump(old + [row], f, indent=1)
        if not a.skip_cpu:
            for mode in CPU_MODES:
                if mode in capped or (mode == 'exact' and nr > a.cpu_exact_max):
                    continue
                tc, wall = cpu_run(mode, a.ndata, nr, a.cap)
                row['cpu_' + mode] = tc
                row['cpu_%s_total' % mode] = sum(tc.values()) if tc else None
                print('  CPU', mode, {k: round(v, 2) for k, v in tc.items()} if tc
                      else f'> {a.cap:.0f}s cap', flush=True)
                if tc is None:
                    capped.add(mode)      # larger N would only take longer
        rows.append(row)
        with open(os.path.join(HERE, 'results_partial.json'), 'w') as f:
            json.dump(rows, f, indent=1)

    vram = None
    if a.vram_test and len(nrs) > 1 and not a.skip_gpu:
        nr = nrs[min(2, len(nrs) - 1)]
        t1, s1, o1, z1, x1 = gpu_run(a.ndata, nr, a.precision)
        t2, s2, o2, z2, x2 = gpu_run(a.ndata, nr, a.precision, max_vram_gb=a.vram_test)
        same = all(np.array_equal(o1[k].ntri if len(k) == 3 else o1[k].npairs,
                                  o2[k].ntri if len(k) == 3 else o2[k].npairs) for k in o1)
        m = np.isfinite(z1)
        vram = dict(nrand=nr, cap_gb=a.vram_test, batches_free=s1['batches'],
                    batches_capped=s2['batches'], peak_free_mib=s1['peak_mib'],
                    peak_capped_mib=s2['peak_mib'], counts_identical=bool(same),
                    max_abs_dzeta=float(np.max(np.abs(z1[m] - z2[m]))),
                    t_free=sum(t1.values()), t_capped=sum(t2.values()))
        print('VRAM cap test', vram, flush=True)

    # ---- output -------------------------------------------------------------------------
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(dict(rows=rows, vram=vram, gpu=gpu_name, cpu=cpu_name, ncpu=ncpu), f, indent=1)
    with open(os.path.join(HERE, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['nrand', 'ndata', 'backend'] + list(TERMS2 + TERMS3) + ['total_s', 'peak_vram_mib',
                                                                         'batches'])
        for r in rows:
            w.writerow([r['nrand'], r['ndata'], 'cuda'] + [f"{r['gpu'][k]:.4f}" for k in TERMS2 + TERMS3]
                       + [f"{r['gpu_total']:.3f}", f"{r['peak_mib']:.0f}", r['batches']])
            for mode in CPU_MODES:
                key = 'cpu_' + mode
                if key in r:
                    t = r[key]
                    w.writerow([r['nrand'], r['ndata'], key] +
                               ([f"{t[k]:.4f}" for k in TERMS2 + TERMS3] if t else ['>cap'] * 7) +
                               [f"{sum(t.values()):.3f}" if t else '>cap', '', ''])

    ns = [r['nrand'] for r in rows]
    fits = {}
    for key in ['gpu_total'] + ['cpu_%s_total' % m for m in CPU_MODES]:
        ts = [r.get(key) if r.get(key) is not None else np.nan for r in rows]
        fits[key] = fit_power(ns, ts)

    def cell(r, key):
        v = r.get(key)
        if v is not None:
            return f'{v:.2f}'
        f = fits.get(key)
        if f is None or key == 'gpu_total':
            return '—'
        return f"~{np.exp(f[1]) * r['nrand'] ** f[0]:.3g} (extrap.)"

    lines = ['# gpucorr benchmark', '',
             f'GPU: {gpu_name} ({gpu_mem:.1f} GiB) · CPU: {cpu_name} ({ncpu} threads, TreeCorr '
             f'OpenMP) · precision: {a.precision} · N_data = {a.ndata} · periodic box L = {L:g}, '
             f'50 log bins 50–150, 50 φ bins', '',
             '## Wall time per term on the GPU (s)', '',
             '| N_rand | ' + ' | '.join(TERMS2 + TERMS3) + ' | total | peak VRAM (MiB) | batches |',
             '|---' * (len(TERMS2 + TERMS3) + 4) + '|']
    for r in rows:
        lines.append(f"| {r['nrand']} | " + ' | '.join(f"{r['gpu'][k]:.2f}" for k in TERMS2 + TERMS3)
                     + f" | {r['gpu_total']:.2f} | {r['peak_mib']:.0f} | {r['batches']} |")
    lines += ['', '## Total wall time (all 7 terms, s) and GPU speed-up', '',
              '| N_rand | GPU exact | ' + ' | '.join(f'{CPU_LABEL[m]} | speed-up' for m in CPU_MODES)
              + ' |', '|---' * (2 + 2 * len(CPU_MODES)) + '|']

    def sp(c, g):
        try:
            return f"{float(c.strip('~').split()[0]) / g:.0f}×" + (' (extrap.)' if '~' in c else '')
        except Exception:
            return '—'
    for r in rows:
        g = r['gpu_total']
        cells = []
        for m in CPU_MODES:
            c = cell(r, 'cpu_%s_total' % m)
            cells += [c, sp(c, g)]
        lines.append(f"| {r['nrand']} | {g:.2f} | " + ' | '.join(cells) + ' |')
    lines += ['', 'Fitted scaling t ∝ N_rand^α: ' + ', '.join(
        f"{k.replace('_total', '')}: α = {v[0]:.2f}" for k, v in fits.items() if v), '']
    if vram:
        lines += ['## Bounded-VRAM run', '',
                  f"N_rand = {vram['nrand']} with `max_vram_gb: {vram['cap_gb']}` → "
                  f"{vram['batches_capped']} batches (vs {vram['batches_free']} uncapped), peak "
                  f"{vram['peak_capped_mib']:.0f} MiB (vs {vram['peak_free_mib']:.0f} MiB); counts "
                  f"identical: **{vram['counts_identical']}**, max |Δzeta| = {vram['max_abs_dzeta']:.1e}; "
                  f"time {vram['t_capped']:.2f} s vs {vram['t_free']:.2f} s.", '']
    with open(os.path.join(HERE, 'results.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    sys.stdout.buffer.write(('\n'.join(lines) + '\n').encode('utf-8'))


if __name__ == '__main__':
    main()
