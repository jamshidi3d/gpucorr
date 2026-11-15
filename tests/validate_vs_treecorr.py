#!/usr/bin/env python
"""Consistency check of the cuda backend (gpucorr) against the CPU reference (TreeCorr).

Part 1  independent NumPy brute force (TreeCorr's leaf formulas, P1-centred orientation)
        vs GPU, bin by bin, Euclidean and Periodic.  Also counts the triangles whose 2<->3
        orientation is ambiguous under TreeCorr's 3D CCW rule (see USAGE.md).
Part 2  TreeCorr exact (bin_slop=0, algo='triangle') vs GPU on random subsamples of the real
        input catalogs: DD, DR, RR -> xi and DDD, DRR, RDD, RRR -> zeta, per config file.
Part 3  TreeCorr *default* settings (bin_slop default, multipole) vs the exact result, to
        quantify the approximation of the CPU default.

Writes tests/validation_report.md (and .json).

    python tests/validate_vs_treecorr.py [--ndata 2000] [--nrand 4000] [--quick]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import treecorr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import gpucorr  # noqa: E402

L = 1000.
MINSEP, MAXSEP, NBINS, NPHI = 50., 150., 50, 50


# ------------------------------------------------------------------------------------------
# Part 1: NumPy brute-force reference
# ------------------------------------------------------------------------------------------
def wrap(d, per):
    if per is None:
        return d
    d = d.copy()
    for _ in range(3):
        d = np.where(d > per / 2., d - per, d)
        d = np.where(d < -per / 2., d + per, d)
    return d


def ref_nnn_auto(p, per):
    """Brute-force LogSAS auto-correlation with TreeCorr's leaf formulas.  Each triangle is
    counted once per vertex as P1; the 2<->3 labelling uses CCW(P1,P3,P2) at P1."""
    n = len(p)
    binsize = np.log(MAXSEP / MINSEP) / NBINS
    phibin = np.pi / NPHI
    logmin = np.log(MINSEP)
    ntri = np.zeros((NBINS, NBINS, NPHI))
    sums = {k: np.zeros((NBINS, NBINS, NPHI)) for k in
            ('d1', 'logd1', 'd2', 'logd2', 'd3', 'logd3', 'phi')}
    namb = 0
    for i in range(n):
        v = wrap(p - p[i], per)
        d = np.sqrt(np.sum(v * v, axis=1))
        ok = (d >= MINSEP) & (d < MAXSEP)
        ok[i] = False
        idx = np.nonzero(ok)[0]
        if len(idx) < 2:
            continue
        jj, ll = np.triu_indices(len(idx), 1)
        j, l = idx[jj], idx[ll]
        vj, vl = v[j], v[l]
        ori = np.einsum('ij,j->i', np.cross(vl, vj), p[i])
        noswap = ori < 0
        P2 = np.where(noswap, j, l)
        P3 = np.where(noswap, l, j)
        v2, v3 = v[P2], v[P3]
        u = wrap(p[P3] - p[P2], per)
        d1sq = np.sum(u * u, axis=1)
        d2sq = np.sum(v3 * v3, axis=1)
        d3sq = np.sum(v2 * v2, axis=1)
        keep = d1sq > 0
        d2, d3 = np.sqrt(d2sq), np.sqrt(d3sq)
        cosphi = (d2sq + d3sq - d1sq) / (2 * d2 * d3)
        phi = np.arccos(np.clip(cosphi, -1, 1))
        phi = np.where(cosphi <= -1, np.pi, np.where(cosphi >= 1, 0., phi))
        keep &= (phi >= 0.) & (phi < np.pi)
        k2 = np.floor((np.log(d2) - logmin) / binsize).astype(int)
        k3 = np.floor((np.log(d3) - logmin) / binsize).astype(int)
        k2 = np.minimum(k2, NBINS - 1)
        k3 = np.minimum(k3, NBINS - 1)
        kp = np.minimum(np.floor(phi / phibin).astype(int), NPHI - 1)
        k2, k3, kp = k2[keep], k3[keep], kp[keep]
        d1 = np.sqrt(d1sq[keep])
        np.add.at(ntri, (k2, k3, kp), 1)
        for name, val in (('d1', d1), ('logd1', np.log(d1)), ('d2', d2[keep]),
                          ('logd2', np.log(d2[keep])), ('d3', d3[keep]),
                          ('logd3', np.log(d3[keep])), ('phi', phi[keep])):
            np.add.at(sums[name], (k2, k3, kp), val)
        # orientation ambiguity: does the CCW test give the same answer at the other two
        # vertices (it should, unless the triangle straddles a periodic boundary)?
        if per is not None:
            pj = p[i] + vj
            pl = p[i] + vl
            nrm = np.cross(vl, vj)
            s0 = np.einsum('ij,j->i', nrm, p[i]) < 0
            s1 = np.einsum('ij,ij->i', nrm, p[j]) < 0
            s2 = np.einsum('ij,ij->i', nrm, p[l]) < 0
            namb += int(np.sum(keep & ((s0 != s1) | (s0 != s2))))
            del pj, pl
    m = ntri > 0
    means = {k: np.where(m, s / np.maximum(ntri, 1), 0.) for k, s in sums.items()}
    return ntri, means, namb


def part1(n, rng):
    out = []
    p = rng.uniform(0, L, (n, 3))
    cat = treecorr.Catalog(x=p[:, 0], y=p[:, 1], z=p[:, 2])
    for metric in ('Euclidean', 'Periodic'):
        per = L if metric == 'Periodic' else None
        t0 = time.perf_counter()
        ntri, means, namb = ref_nnn_auto(p, per)
        tref = time.perf_counter() - t0
        for prec in ('mixed', 'double'):
            kw = dict(min_sep=MINSEP, max_sep=MAXSEP, nbins=NBINS, nphi_bins=NPHI,
                      bin_type='LogSAS')
            if per:
                kw['period'] = per
            g = gpucorr.GPUNNNCorrelation(gpu_opts=gpucorr.GPUOptions(gpu_precision=prec), **kw)
            g.process(cat, metric=metric)
            m = ntri > 0
            rel = {}
            for k, a in (('d1', g.meand1), ('logd1', g.meanlogd1), ('d2', g.meand2),
                         ('logd2', g.meanlogd2), ('d3', g.meand3), ('logd3', g.meanlogd3),
                         ('phi', g.meanphi)):
                rel[k] = float(np.max(np.abs(a[m] - means[k][m]) / np.abs(means[k][m])))
            out.append(dict(metric=metric, precision=prec, n=n, ntri=float(ntri.sum()),
                            bins_differ=int(np.sum(ntri != g.ntri)),
                            max_abs_dntri=float(np.max(np.abs(ntri - g.ntri))),
                            max_rel_dmean=rel, ambiguous_rotations=namb,
                            exact_path=g.gpu_stats.nexact, t_ref=tref))
    return out


# ------------------------------------------------------------------------------------------
# Part 2/3: TreeCorr vs GPU on the real catalogs
# ------------------------------------------------------------------------------------------
def load_sub(nd, nr, rng):
    data = np.loadtxt(os.path.join(ROOT, 'input', 'data.data'))
    rand = np.loadtxt(os.path.join(ROOT, 'input', 'random.txt'))
    d = data[rng.choice(len(data), nd, replace=False)] if nd < len(data) else data
    r = rand[rng.choice(len(rand), nr, replace=False)] if nr < len(rand) else rand
    dcat = treecorr.Catalog(x=d[:, 0], y=d[:, 1], z=d[:, 2], w=d[:, 3] if d.shape[1] > 3 else None)
    rcat = treecorr.Catalog(x=r[:, 0], y=r[:, 1], z=r[:, 2])
    return dcat, rcat


def nn_terms(cls, dcat, rcat, metric, **kw):
    t = {}
    tm = {}
    for name, args in (('DD', (dcat,)), ('DR', (dcat, rcat)), ('RR', (rcat,))):
        c = cls(**kw)
        t0 = time.perf_counter()
        c.process(*args, metric=metric)
        tm[name] = time.perf_counter() - t0
        t[name] = c
    xi, varxi = t['DD'].calculateXi(rr=t['RR'], dr=t['DR'])
    return t, xi, tm


def nnn_terms(cls, dcat, rcat, metric, algo=None, **kw):
    t = {}
    tm = {}
    for name, args, ordered in (('DDD', (dcat,), True), ('DRR', (dcat, rcat), False),
                                ('RDD', (rcat, dcat), False), ('RRR', (rcat,), True)):
        c = cls(**kw)
        t0 = time.perf_counter()
        pk = dict(metric=metric)
        if len(args) == 2:
            pk['ordered'] = ordered
        if algo is not None:
            pk['algo'] = algo
        c.process(*args, **pk)
        tm[name] = time.perf_counter() - t0
        t[name] = c
    zeta, varzeta = t['DDD'].calculateZeta(rrr=t['RRR'], drr=t['DRR'], rdd=t['RDD'])
    return t, zeta, tm


def cmp_nn(a, b):
    r = {}
    for k in a:
        x, y = a[k], b[k]
        m = x.weight > 0
        r[k] = dict(max_abs_dnpairs=float(np.max(np.abs(x.npairs - y.npairs))),
                    max_rel_dweight=float(np.max(np.abs(x.weight[m] - y.weight[m]) / x.weight[m])),
                    max_rel_dmeanr=float(np.max(np.abs(x.meanr[m] - y.meanr[m]) / x.meanr[m])),
                    max_abs_dmeanlogr=float(np.max(np.abs(x.meanlogr[m] - y.meanlogr[m]))))
    return r


def sym(a):
    return a + a.transpose(1, 0, 2)


def cmp_nnn(a, b):
    r = {}
    for k in a:
        x, y = a[k], b[k]
        m = (x.ntri == y.ntri) & (x.weight > 0)
        d = x.ntri - y.ntri
        r[k] = dict(total_ntri_ref=float(x.ntri.sum()), total_ntri_gpu=float(y.ntri.sum()),
                    bins_differ=int(np.sum(d != 0)), sum_abs_dntri=float(np.abs(d).sum()),
                    frac_moved=float(np.abs(d).sum() / 2 / max(x.ntri.sum(), 1)),
                    sym_bins_differ=int(np.sum(sym(x.ntri) != sym(y.ntri))),
                    phi_marginal_maxdiff=float(np.max(np.abs(x.ntri.sum((0, 1)) - y.ntri.sum((0, 1))))),
                    max_rel_dmeand1_equal_bins=float(np.max(np.abs(x.meand1[m] - y.meand1[m]) / x.meand1[m])) if m.any() else 0.,
                    max_rel_dmeanphi_equal_bins=float(np.max(np.abs(x.meanphi[m] - y.meanphi[m]) / x.meanphi[m])) if m.any() else 0.)
    return r


def zeta_from(t, f=lambda a: a):
    """Full estimator from (optionally transformed) weights, exactly as calculateZeta."""
    ddd, drr, rdd, rrr = (f(t[k].weight) for k in ('DDD', 'DRR', 'RDD', 'RRR'))
    rrrf = t['DDD'].tot / t['RRR'].tot
    drrf = t['DDD'].tot / t['DRR'].tot
    rddf = t['DDD'].tot / t['RDD'].tot
    den = rrr * rrrf
    with np.errstate(divide='ignore', invalid='ignore'):
        return (ddd - rdd * rddf + drr * drrf - rrr * rrrf) / den, den > 0


def zeta_cmp(z1, z2, t1, t2):
    m = t1['RRR'].weight > 0
    s1, m1 = zeta_from(t1, sym)
    s2, m2 = zeta_from(t2, sym)
    mm = m1 & m2
    good = t1['RRR'].weight >= 100      # bins with a usable number of random triangles
    return dict(max_abs_dzeta=float(np.max(np.abs(z1[m] - z2[m]))),
                max_abs_dzeta_good=float(np.max(np.abs(z1[good] - z2[good]))) if good.any() else 0.,
                rms_zeta=float(np.sqrt(np.mean(z1[m] ** 2))),
                max_abs_dzeta_symmetrised=float(np.max(np.abs(s1[mm] - s2[mm]))))


def part2(nd, nr, rng, metrics, prec):
    dcat, rcat = load_sub(nd, nr, rng)
    res = {'ndata': nd, 'nrand': nr}
    kw2 = dict(min_sep=MINSEP, max_sep=MAXSEP, nbins=NBINS)
    kw3 = dict(kw2, nphi_bins=NPHI, min_phi=0., max_phi=np.pi, bin_type='LogSAS')
    gopt = gpucorr.GPUOptions(gpu_precision=prec)
    for metric in metrics:
        k2 = dict(kw2, period=L) if metric == 'Periodic' else kw2
        k3 = dict(kw3, period=L) if metric == 'Periodic' else kw3
        print(f'[{metric}] 2pt ...', flush=True)
        ex, xi_ex, tm_ex = nn_terms(treecorr.NNCorrelation, dcat, rcat, metric, bin_slop=0, **k2)
        de, xi_de, tm_de = nn_terms(treecorr.NNCorrelation, dcat, rcat, metric, **k2)
        gp, xi_gp, tm_gp = nn_terms(lambda **k: gpucorr.GPUNNCorrelation(gpu_opts=gopt, **k),
                                    dcat, rcat, metric, **k2)
        m = np.isfinite(xi_ex)
        r2 = dict(gpu_vs_exact=cmp_nn(ex, gp),
                  max_abs_dxi_gpu_vs_exact=float(np.max(np.abs(xi_ex[m] - xi_gp[m]))),
                  max_abs_dxi_default_vs_exact=float(np.max(np.abs(xi_ex[m] - xi_de[m]))),
                  max_rel_dRR_default_vs_exact=float(np.max(np.abs(ex['RR'].weight - de['RR'].weight) / ex['RR'].weight)),
                  times=dict(cpu_exact=tm_ex, cpu_default=tm_de, gpu=tm_gp))
        print(f'[{metric}] 3pt GPU ...', flush=True)
        gp3, z_gp, tm_gp3 = nnn_terms(lambda **k: gpucorr.GPUNNNCorrelation(gpu_opts=gopt, **k),
                                      dcat, rcat, metric, **k3)
        print(f'[{metric}] 3pt CPU exact (bin_slop=0, triangle) ...', flush=True)
        ex3, z_ex, tm_ex3 = nnn_terms(treecorr.NNNCorrelation, dcat, rcat, metric,
                                      algo='triangle', bin_slop=0, **k3)
        print(f'[{metric}] 3pt CPU default (multipole) ...', flush=True)
        de3, z_de, tm_de3 = nnn_terms(treecorr.NNNCorrelation, dcat, rcat, metric, **k3)
        r3 = dict(gpu_vs_exact=cmp_nnn(ex3, gp3),
                  zeta_gpu_vs_exact=zeta_cmp(z_ex, z_gp, ex3, gp3),
                  zeta_default_vs_exact=zeta_cmp(z_ex, z_de, ex3, de3),
                  RRR_default_vs_exact_max_rel=float(np.max(
                      np.abs(ex3['RRR'].weight - de3['RRR'].weight)[ex3['RRR'].weight > 0]
                      / ex3['RRR'].weight[ex3['RRR'].weight > 0])),
                  times=dict(cpu_exact=tm_ex3, cpu_default=tm_de3, gpu=tm_gp3),
                  gpu_exact_path=int(sum(c.gpu_stats.nexact for c in gp3.values())))
        res[metric] = dict(nn=r2, nnn=r3)
    return res


def part3(rng):
    """TreeCorr default LogSAS algorithm (multipole) vs its triangle algorithm, 2D vs 3D."""
    out = []
    kw = dict(min_sep=MINSEP, max_sep=MAXSEP, nbins=NBINS, nphi_bins=NPHI, bin_type='LogSAS')
    for dim, n in ((2, 6000), (3, 20000)):
        p = rng.uniform(0, L, (n, dim))
        cat = treecorr.Catalog(x=p[:, 0], y=p[:, 1], z=p[:, 2] if dim == 3 else None)
        tr = treecorr.NNNCorrelation(**kw)
        tr.process(cat, algo='triangle')
        mp = treecorr.NNNCorrelation(**kw)
        mp.process(cat)
        ref = tr
        if dim == 3:
            ex = gpucorr.GPUNNNCorrelation(**kw)
            ex.process(cat)
            ref = ex
        m = ref.weight > 300
        r_mp = np.abs(mp.weight[m] / ref.weight[m] - 1)
        r_tr = np.abs(tr.weight[m] / ref.weight[m] - 1)
        out.append(dict(dim=dim, n=n, ref='GPU exact' if dim == 3 else 'TreeCorr triangle',
                        mp_median=float(np.median(r_mp)), mp_max=float(r_mp.max()),
                        mp_phimarg=float(np.max(np.abs(mp.weight.sum((0, 1)) / ref.weight.sum((0, 1)) - 1))),
                        tr_median=float(np.median(r_tr)), tr_max=float(r_tr.max())))
    return out


def fmt(x):
    return f'{x:.3g}' if isinstance(x, float) else str(x)


def write_report(p1, p2, path, p3=None):
    lines = ['# gpucorr vs TreeCorr — consistency report', '',
             f'TreeCorr {treecorr.__version__}, gpucorr {gpucorr.__version__}, '
             f'binning: {NBINS} log bins {MINSEP}–{MAXSEP}, {NPHI} phi bins [0, π].', '']
    lines += ['## Part 1 — independent NumPy brute force vs GPU (auto NNN, uniform points)', '',
              '| metric | precision | N | triangles (rotations) | bins differing | max abs Δntri | '
              'max rel Δ meand1 | Δ meanlogd1 | Δ meand2 | Δ meanphi | exact-path triplets | '
              'orientation-ambiguous rotations |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in p1:
        mr = r['max_rel_dmean']
        lines.append(f"| {r['metric']} | {r['precision']} | {r['n']} | {r['ntri']:.0f} | "
                     f"{r['bins_differ']} | {r['max_abs_dntri']:.0f} | {mr['d1']:.2e} | "
                     f"{mr['logd1']:.2e} | {mr['d2']:.2e} | {mr['phi']:.2e} | {r['exact_path']} | "
                     f"{r['ambiguous_rotations']} |")
    lines.append('')
    if p2:
        lines += [f"## Part 2 — TreeCorr exact (bin_slop=0, algo='triangle') vs GPU, real "
                  f"catalogs (subsample: {p2['ndata']} data, {p2['nrand']} randoms)", '']
        for metric in ('Euclidean', 'Periodic'):
            if metric not in p2:
                continue
            r = p2[metric]
            lines += [f'### {metric}', '', '**2-point (NN)**', '',
                      '| term | max abs Δnpairs | max rel Δweight | max rel Δmeanr | max abs Δmeanlogr |',
                      '|---|---|---|---|---|']
            for k, v in r['nn']['gpu_vs_exact'].items():
                lines.append(f"| {k} | {v['max_abs_dnpairs']:.0f} | {v['max_rel_dweight']:.2e} | "
                             f"{v['max_rel_dmeanr']:.2e} | {v['max_abs_dmeanlogr']:.2e} |")
            lines += ['', f"max |Δxi| GPU vs exact: **{r['nn']['max_abs_dxi_gpu_vs_exact']:.2e}**; "
                          f"TreeCorr default (bin_slop) vs exact: {r['nn']['max_abs_dxi_default_vs_exact']:.2e} "
                          f"(max rel ΔRR {r['nn']['max_rel_dRR_default_vs_exact']:.2e}).", '',
                      '**3-point (NNN, LogSAS)**', '',
                      '| term | total ntri exact | total ntri GPU | bins differing | Σ\\|Δntri\\| | '
                      'fraction moved (d2↔d3) | bins differing after d2↔d3 symmetrisation | '
                      'φ-marginal max Δ | max rel Δmeand1 (equal-count bins) |',
                      '|---|---|---|---|---|---|---|---|---|']
            for k, v in r['nnn']['gpu_vs_exact'].items():
                lines.append(f"| {k} | {v['total_ntri_ref']:.0f} | {v['total_ntri_gpu']:.0f} | "
                             f"{v['bins_differ']} | {v['sum_abs_dntri']:.0f} | {v['frac_moved']:.2e} | "
                             f"{v['sym_bins_differ']} | {v['phi_marginal_maxdiff']:.0f} | "
                             f"{v['max_rel_dmeand1_equal_bins']:.2e} |")
            zg, zd = r['nnn']['zeta_gpu_vs_exact'], r['nnn']['zeta_default_vs_exact']
            lines += ['', f"zeta, full estimator (DDD−RDD+DRR−RRR)/RRR, rms over bins = {zg['rms_zeta']:.3g}:", '',
                      f"* GPU vs TreeCorr exact: max |Δzeta| = **{zg['max_abs_dzeta']:.2e}** over all bins "
                      f"({zg['max_abs_dzeta_good']:.2e} over bins with RRR ≥ 100); with the counts "
                      f"symmetrised over d2↔d3 before forming zeta: **{zg['max_abs_dzeta_symmetrised']:.2e}**.",
                      f"* TreeCorr *default* (multipole) vs TreeCorr exact: max |Δzeta| = {zd['max_abs_dzeta']:.2e} "
                      f"(symmetrised {zd['max_abs_dzeta_symmetrised']:.2e}); max rel ΔRRR = "
                      f"{r['nnn']['RRR_default_vs_exact_max_rel']:.2e}.", '']
            t = r['nnn']['times']
            lines += ['Timings (s): ' + ', '.join(
                f"{k}: CPU exact {t['cpu_exact'][k]:.2f} / CPU default {t['cpu_default'][k]:.2f} / GPU {t['gpu'][k]:.2f}"
                for k in t['gpu']), '']
    if p3:
        lines += ['## Part 3 — TreeCorr default LogSAS algorithm (multipole) in 2D vs 3D', '',
                  'Uniform points, same binning, auto-correlation; bins with > 300 triangles.', '',
                  '| dim | N | reference | multipole: median rel. Δ | multipole: max rel. Δ | '
                  'multipole: φ-marginal max rel. Δ | triangle (default bin_slop): median | max |',
                  '|---|---|---|---|---|---|---|---|']
        for r in p3:
            lines.append(f"| {r['dim']}D | {r['n']} | {r['ref']} | {r['mp_median']:.2e} | {r['mp_max']:.2e} | "
                         f"{r['mp_phimarg']:.2e} | {r['tr_median']:.2e} | {r['tr_max']:.2e} |")
        lines += ['', 'In 2D the multipole algorithm agrees with the triangle algorithm at the per-cent level. '
                  'For 3D (x, y, z) catalogs it does not reproduce the opening-angle distribution, whereas '
                  'the TreeCorr triangle algorithm and gpucorr agree.', '']
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n1', type=int, default=2000, help='points for the NumPy brute-force part')
    ap.add_argument('--ndata', type=int, default=2000)
    ap.add_argument('--nrand', type=int, default=4000)
    ap.add_argument('--precision', default='mixed')
    ap.add_argument('--metrics', default='Euclidean,Periodic')
    ap.add_argument('--skip2', action='store_true')
    a = ap.parse_args()
    rng = np.random.default_rng(12345)
    print('Part 1 ...', flush=True)
    p1 = part1(a.n1, rng)
    for r in p1:
        print(r, flush=True)
    p2 = None
    if not a.skip2:
        p2 = part2(a.ndata, a.nrand, rng, a.metrics.split(','), a.precision)
    print('Part 3 ...', flush=True)
    p3 = part3(rng)
    out = os.path.join(HERE, 'validation_report')
    with open(out + '.json', 'w') as f:
        json.dump(dict(part1=p1, part2=p2, part3=p3), f, indent=1)
    write_report(p1, p2, out + '.md', p3)
    print('wrote', out + '.md')


if __name__ == '__main__':
    main()
