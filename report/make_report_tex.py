#!/usr/bin/env python
"""Write report/gpucorr_vs_treecorr.tex (+ vector figures) from the result files and compile it.

Inputs: tests/validation_report.json, bench/results.json, report/legendre_results.json,
        report/legendre_binning_test.json
    python report/make_report_tex.py            # writes .tex, figures, runs latexmk
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import figures as MR  # noqa: E402  (figure helpers, extrapolation)

TERMS = MR.TERMS
T3 = ('DDD', 'DRR', 'RDD', 'RRR')


# ------------------------------------------------------------------------------------------
# formatting helpers
# ------------------------------------------------------------------------------------------
def num(x, d=2):
    """Number in siunitx form, e.g. 1.2e-07 -> \\num{1.2e-7}."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '---'
    if x == 0:
        return r'\num{0}'
    if abs(x) >= 1e-2 and abs(x) < 1e5:
        return r'\num{%s}' % (f'{x:.{d}f}')
    return r'\num{%s}' % (f'{x:.{max(d - 1, 0)}e}')


def sec(t):
    if t is None:
        return '---'
    if t < 120:
        return r'\SI{%.2f}{s}' % t if t < 10 else r'\SI{%.1f}{s}' % t
    if t < 7200:
        return r'\SI{%.1f}{min}' % (t / 60)
    return r'\SI{%.1f}{h}' % (t / 3600)


def table(header, rows, spec, caption=None, label=None, small=True):
    out = [r'\begin{table}[htbp]', r'\centering']
    if small:
        out.append(r'\small')
    if caption:
        out.append(r'\caption{%s}' % caption)
    if label:
        out.append(r'\label{%s}' % label)
    out += [r'\begin{tabular}{%s}' % spec, r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    out += [' & '.join(str(c) for c in r) + r' \\' for r in rows]
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


def figure(path, caption, label, width=r'0.92\textwidth'):
    return '\n'.join([r'\begin{figure}[htbp]', r'\centering',
                      r'\includegraphics[width=%s]{%s}' % (width, os.path.basename(path)),
                      r'\caption{%s}' % caption, r'\label{%s}' % label, r'\end{figure}'])


# ------------------------------------------------------------------------------------------
def main():
    val, bench = MR.load()
    rows = bench['rows']
    leg_path = os.path.join(HERE, 'legendre_results.json')
    leg = json.load(open(leg_path)) if os.path.exists(leg_path) else None
    bt = json.load(open(os.path.join(HERE, 'legendre_binning_test.json')))

    # vector figures
    f_scal = os.path.join(HERE, 'fig_scaling.pdf')
    f_terms = os.path.join(HERE, 'fig_terms.pdf')
    f_mp = os.path.join(HERE, 'fig_multipole.pdf')
    f_lt = os.path.join(HERE, 'fig_leg_timing.pdf')
    MR.fig_scaling(rows, f_scal)
    MR.fig_terms(rows, f_terms)
    mpinfo = MR.fig_multipole(f_mp)
    if leg and leg.get('timing'):
        MR.fig_leg_timing(leg['timing'], f_lt)

    p1, p2, p3 = val['part1'], val['part2'], val.get('part3') or []
    last = rows[-1]

    def tot(r, m):
        return r.get('cpu_%s_total' % m)

    def cpu(r, m):
        v = tot(r, m)
        return (v, False) if v is not None else (MR.cpu_extrap(rows, m, r['nrand']), True)

    bt500, _ = cpu(last, 'triangle')
    eu, pe = p2['Euclidean'], p2['Periodic']
    gpu_name = bench['gpu']
    V = {}

    # ---- abstract / summary -----------------------------------------------------------------
    leg_sum = ''
    if leg and leg.get('timing'):
        r = [x for x in leg['timing'] if x.get('leg10') and x.get('exact')][-1]
        t10 = sum(r['leg10'][k] for k in T3)
        tex = sum(r['exact'][k] for k in T3)
        leg_sum = (r'\item \textbf{Legendre mode (optional).} \texttt{gpu\_method: legendre} computes the '
                   r'Slepian--Eisenstein multipoles exactly (\num{4e-16} agreement with brute force). At '
                   r'$N_\mathrm{rand}=\num{%d}$ the 3PCF takes %s with $\ell_\mathrm{max}=10$ instead of %s for '
                   r'exact binned counting (%.0f$\times$ faster); $\phi$-binned values reconstructed from it agree '
                   r'with exact within the shot noise (Sec.~\ref{sec:legendre}).' % (r['nrand'], sec(t10), sec(tex), tex / t10))
    V['SUMMARY'] = '\n'.join([
        r'\begin{itemize}',
        r'\item \textbf{Speed.} The complete estimator set (DD, DR, RR, DDD, DRR, RDD, RRR) with '
        r'$N_\mathrm{data}=\num{%d}$ and $N_\mathrm{rand}=\num{%d}$ runs in \textbf{%s} on the GPU. TreeCorr\textquotesingle s '
        r'accurate tree algorithm (\texttt{triangle}, default \texttt{bin\_slop}) would need $\sim$%s (extrapolated), '
        r'about %.0f$\times$ longer; exact TreeCorr (\texttt{bin\_slop=0}) costs the same. TreeCorr\textquotesingle s default '
        r'\texttt{multipole} algorithm is 4--14$\times$ slower than the GPU up to $N_\mathrm{rand}=\num{200000}$ and about as '
        r'fast at \num{500000}, but it is not correct in 3D (next items).'
        % (last['ndata'], last['nrand'], sec(last['gpu_total']), sec(bt500), bt500 / last['gpu_total']),
        r'\item \textbf{Precision.} The GPU counts every pair and triangle. Against exact TreeCorr '
        r'(\texttt{bin\_slop=0}, \texttt{algo=triangle}) on the real catalogs every term agrees bin by bin in the '
        r'Euclidean metric ($\Delta\xi=%s$, $\Delta\zeta=%s$); mean columns agree to $\sim\num{1e-6}$ (mixed precision) '
        r'or \numrange{1e-12}{1e-15} (double precision).'
        % (num(eu['nn']['max_abs_dxi_gpu_vs_exact']), num(eu['nnn']['zeta_gpu_vs_exact']['max_abs_dzeta'])),
        r'\item \textbf{Periodic box.} Totals, $\phi$-marginals and $d_2\leftrightarrow d_3$-symmetrised counts are '
        r'identical; a fraction $\le %s$ of triangle rotations sits in the mirror bin $(d_3,d_2)$ because '
        r'TreeCorr\textquotesingle s 3D orientation rule is ambiguous for triangles crossing the box edge '
        r'(Sec.~\ref{sec:orient}).' % num(max(v['frac_moved'] for v in pe['nnn']['gpu_vs_exact'].values()), 1),
        r'\item \textbf{TreeCorr default in 3D.} The default LogSAS algorithm (\texttt{multipole}) does not reproduce '
        r'the 3D opening-angle distribution: median per-bin error %.0f\,\%% versus %.2f\,\%% for the \texttt{triangle} '
        r'algorithm (Sec.~\ref{sec:mp3d}).' % (100 * mpinfo['mp_med'], 100 * mpinfo['tr_med']),
        leg_sum,
        r'\end{itemize}'])

    # ---- setup table --------------------------------------------------------------------------
    V['SETUP_TABLE'] = table(
        ['Backend / method', 'Algorithm', 'Approximation'],
        [[r'GPU, exact (\texttt{backend: cuda})', r'periodic cell list; every candidate tested with TreeCorr\textquotesingle s '
          r'leaf formulas; fp32 geometry with fp64 recheck near bin edges (mixed) or all fp64 (double)', 'none'],
         [r'GPU, Legendre (\texttt{gpu\_method: legendre})', r'spherical-harmonic multipoles $a_{\ell m}$ + cuBLAS '
          r'products (Slepian \& Eisenstein 2015)', r'none for $\ell\le\ell_\mathrm{max}$; $\phi$ bins truncated'],
         ['CPU, TreeCorr default', r'ball tree, \texttt{bin\_slop}$\approx1$ (2PCF); \texttt{multipole} (3PCF)',
          r'cells accepted whole; incorrect $\phi$ in 3D'],
         ['CPU, TreeCorr triangle', r'ball tree, default \texttt{bin\_slop}, \texttt{triangle}', r'cells accepted whole ($\sim$0.1\,\%)'],
         ['CPU, TreeCorr exact', r'ball tree, \texttt{bin\_slop=0}, \texttt{triangle}', 'none (scientific reference)']],
        r'>{\raggedright\arraybackslash}p{0.25\textwidth}>{\raggedright\arraybackslash}p{0.45\textwidth}'
        r'>{\raggedright\arraybackslash}p{0.2\textwidth}', caption='Compared methods.', label='tab:methods')

    # ---- speed tables -------------------------------------------------------------------------
    srows = []
    for r in rows:
        line = [r'\num{%d}' % r['nrand'], sec(r['gpu_total'])]
        for m in ('default', 'triangle', 'exact'):
            v, isx = cpu(r, m)
            line.append(('$\\sim$' + sec(v) + r'$^\ast$') if isx and v else sec(v))
        v, isx = cpu(r, 'triangle')
        line.append(('$\\sim$%.0f$\\times$' if isx else '%.0f$\\times$') % (v / r['gpu_total']))
        srows.append(line)
    V['SPEED_TABLE'] = table(
        [r'$N_\mathrm{rand}$', 'GPU exact', 'CPU default', 'CPU triangle', 'CPU exact', 'speed-up'],
        srows, 'rrrrrr',
        caption=r'Wall time of all seven terms ($N_\mathrm{data}=\num{%d}$). $^\ast$Exceeded the \SI{600}{s} CPU cap: '
                r'extrapolated term by term with power laws fitted to the measured sizes (two sizes for the '
                r'triangle/exact modes, so order-of-magnitude). Speed-up is relative to CPU triangle.' % rows[0]['ndata'],
        label='tab:speed')
    trows = [[r'\num{%d}' % r['nrand']] + [r'\num{%.2f}' % r['gpu'][k] for k in TERMS] +
             [num(r['gpu_total'], 1), r'\num{%.0f}' % r['peak_mib'], str(r['batches'])] for r in rows]
    V['TERMS_TABLE'] = table([r'$N_\mathrm{rand}$'] + list(TERMS) + ['total', 'VRAM [MiB]', 'batches'], trows,
                             'r' * 11, caption=r'GPU time per term in seconds, peak memory and number of memory '
                             r'batches. The \num{200000} peak was recorded before a memory-budget fix; with the fix '
                             r'RRR peaks at \SI{4.2}{GB} with unchanged time.', label='tab:terms')
    v = bench.get('vram') or {}
    V['VRAM'] = (r'Re-running $N_\mathrm{rand}=\num{%d}$ with \texttt{max\_vram\_gb: %s} used %d batches instead of '
                 r'%d and peaked at \SI{%.0f}{MiB} instead of \SI{%.0f}{MiB}; all counts were identical and '
                 r'$\max|\Delta\zeta|=%s$ (%s vs %s). Results are independent of the memory budget, so the same '
                 r'run fits a \SI{12}{GB} Colab GPU or any smaller card.'
                 % (v['nrand'], v['cap_gb'], v['batches_capped'], v['batches_free'], v['peak_capped_mib'],
                    v['peak_free_mib'], num(v['max_abs_dzeta']), sec(v['t_capped']), sec(v['t_free']))) if v else ''

    # ---- precision tables ---------------------------------------------------------------------
    prow = []
    for r in p1:
        m = r['max_rel_dmean']
        prow.append([r['metric'], r['precision'], r'\num{%d}' % r['ntri'], str(r['bins_differ']), num(m['d1'], 2),
                     num(m['d2'], 2), num(m['phi'], 2), r'\num{%d}' % r['exact_path']])
    V['P1_TABLE'] = table(['metric', 'precision', 'rotations', 'bins differ', r'$\Delta\bar d_1$',
                           r'$\Delta\bar d_2$', r'$\Delta\bar\phi$', 'fp64 rechecks'], prow, 'llrrrrrr',
                          caption=r'GPU vs an independent NumPy brute force (2,000 uniform points, auto NNN). '
                                  r'$\Delta$ = maximum relative difference of the bin means.', label='tab:p1')

    def p2tab(metric):
        nn, nnn = p2[metric]['nn'], p2[metric]['nnn']
        rr = []
        for k in ('DD', 'DR', 'RR'):
            x = nn['gpu_vs_exact'][k]
            rr.append([k, '%.0f' % x['max_abs_dnpairs'], num(x['max_rel_dweight']), num(x['max_rel_dmeanr']),
                       '---', '---', '---'])
        for k in T3:
            x = nnn['gpu_vs_exact'][k]
            rr.append([k, '%.0f' % abs(x['total_ntri_ref'] - x['total_ntri_gpu']), '---', '---',
                       str(x['bins_differ']), '%.0f' % x['sum_abs_dntri'], str(x['sym_bins_differ'])])
        zg, zd = nnn['zeta_gpu_vs_exact'], nnn['zeta_default_vs_exact']
        cap = (r'%s metric: GPU vs exact TreeCorr (subsample %s data, %s randoms). For NNN the count column is the '
               r'difference of the totals. $\max|\Delta\xi|$: GPU %s, TreeCorr default %s. $\max|\Delta\zeta|$ '
               r'(bins with RRR $\ge100$): GPU %s; from $d_2\leftrightarrow d_3$-symmetrised counts %s; TreeCorr '
               r'default (multipole) %s.' % (metric, r'\num{%d}' % p2['ndata'], r'\num{%d}' % p2['nrand'],
                                             num(nn['max_abs_dxi_gpu_vs_exact']), num(nn['max_abs_dxi_default_vs_exact']),
                                             num(zg['max_abs_dzeta_good']), num(zg['max_abs_dzeta_symmetrised']),
                                             num(zd['max_abs_dzeta_good'])))
        return table(['term', r'$|\Delta$count$|$', r'$\Delta w$ (rel)', r'$\Delta\bar r$ (rel)', 'bins differ',
                      r'$\sum|\Delta n_\mathrm{tri}|$', r'sym.\ bins differ'], rr, 'lrrrrrr', caption=cap,
                     label='tab:p2%s' % metric[:3].lower())
    V['P2E_TABLE'] = p2tab('Euclidean')
    V['P2P_TABLE'] = p2tab('Periodic')
    V['AMBIG'] = r'\num{%d} of \num{%d}' % (p1[2]['ambiguous_rotations'], p1[2]['ntri'])

    mrows = [[r'%dD, $N=\num{%d}$' % (r['dim'], r['n']), r['ref'], num(r['mp_median']), num(r['mp_phimarg']),
              '(reference)' if r['dim'] == 2 else num(r['tr_median'])] for r in p3]
    V['MP_TABLE'] = table(['catalog', 'reference', r'multipole: median $|\Delta|$', r'multipole: $\phi$-marg.\ max',
                           r'triangle: median $|\Delta|$'], mrows, 'llrrr',
                          caption=r'TreeCorr\textquotesingle s default LogSAS algorithm in 2D and 3D (relative '
                                  r'differences, bins with $>300$ triangles).', label='tab:mp')
    V['MP_TEXT'] = (r'In 3D the median per-bin deviation of the multipole algorithm from exact counting is %.0f\,\%% '
                    r'(triangle algorithm: %.2f\,\%%), while the total over all bins is still correct (%s). Raising '
                    r'\texttt{max\_n} to 200 does not change it, so this is not truncation.'
                    % (100 * mpinfo['mp_med'], 100 * mpinfo['tr_med'], num(mpinfo['mp_tot'])))

    # ---- Legendre section ---------------------------------------------------------------------
    if leg:
        acc = leg['accuracy']
        lr = [[str(L), num(acc['rrr_med_rel'][i]), num(acc['rrr_phimarg_max_rel'][i]), num(acc['ddd_med_rel'][i]),
               num(acc['zeta_rms_abs'][i], 3), num(acc['zeta_rms_over_sigma'][i], 2)] for i, L in enumerate(acc['lmax'])]
        V['LEG_BIN_TABLE'] = table([r'$\ell_\mathrm{max}$', r'RRR median $|\Delta|$', r'RRR $\phi$-marg.\ max',
                                    r'DDD median $|\Delta|$', r'rms $\Delta\zeta$', r'rms $\Delta\zeta/\sigma_\zeta$'],
                                   lr, 'rrrrrr', caption=r'$\phi$-binned reconstruction from the Legendre mode vs '
                                   r'symmetrised exact counts (\num{%d} data, \num{%d} randoms, periodic box). '
                                   r'$\sigma_\zeta$: shot-noise error of the exact $\zeta$ (median %s; rms $\zeta$ = %s).'
                                   % (acc['ndata'], acc['nrand'], num(acc['sigma_median'], 3), num(acc['zeta_rms'], 3)),
                                   label='tab:legbin')
        ls = acc['multipole_l']
        e1 = lambda x: r'\num{%.1e}' % x if x else r'\num{0}'
        b200 = dict(zip(bt['l'], bt['nphi200']))
        V['LEG_ML_TABLE'] = table(
            [r'$\ell$', r'rms $\zeta_\ell$', r'rel.\ $\Delta$ (50 $\phi$ bins)', r'rel.\ $\Delta$ (200 $\phi$ bins)'],
            [[str(l), e1(acc['zeta_l_rms'][i]), e1(acc['zeta_l_rel_diff'][i]), e1(b200[l]) if l in b200 else '---']
             for i, l in enumerate(ls)],
            'rrrr', caption=r'Multipoles $\zeta_\ell$: Legendre mode vs projection of the exact '
            r'$\phi$-binned counts onto $P_\ell$ (rms over $(d_2,d_3)$ bins, relative to rms $\zeta_\ell$). The '
            r'difference shrinks as the exact binning is refined.', label='tab:legml')
        V['LEG_MONO'] = num(max(acc['mono_rel'].values()), 1)
        tl = []
        for r in leg.get('timing', []):
            te = sum(r['exact'][k] for k in T3) if r.get('exact') else None
            t10 = sum(r['leg10'][k] for k in T3) if r.get('leg10') else None
            t49 = sum(r['leg49'][k] for k in T3) if r.get('leg49') else None
            tl.append([r'\num{%d}' % r['nrand'], sec(te), sec(t10), ('%.1f$\\times$' % (te / t10)) if te and t10 else '---',
                       sec(t49), ('%.1f$\\times$' % (te / t49)) if te and t49 else '---'])
        V['LEG_TIME_TABLE'] = table([r'$N_\mathrm{rand}$', 'exact', r'Legendre $\ell_\mathrm{max}=10$', 'speed-up',
                                     r'Legendre $\ell_\mathrm{max}=49$', 'speed-up'], tl, 'rrrrrr',
                                    caption='3PCF wall time (DDD, DRR, RDD, RRR) of the two GPU methods.',
                                    label='tab:legtime')
    V['GPU'] = gpu_name.replace('_', r'\_')
    V['CPU'] = r'%s, %d threads' % (bench['cpu'].replace('_', r'\_'), bench['ncpu'])
    V['TCVER'] = val.get('treecorr', '5.1.4')

    tpl = open(os.path.join(HERE, 'report_template.tex'), encoding='utf-8').read()
    for k, v in V.items():
        tpl = tpl.replace('<<%s>>' % k, v)
    if not leg:
        tpl = tpl.split('%%LEG-BEGIN')[0] + tpl.split('%%LEG-END')[1]
    out = os.path.join(HERE, 'gpucorr_vs_treecorr.tex')
    open(out, 'w', encoding='utf-8').write(tpl)
    print('wrote', out)
    subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'gpucorr_vs_treecorr.tex'],
                   cwd=HERE, check=True, capture_output=True)
    subprocess.run(['latexmk', '-c', 'gpucorr_vs_treecorr.tex'], cwd=HERE, capture_output=True)
    print('compiled', os.path.join(HERE, 'gpucorr_vs_treecorr.pdf'))


if __name__ == '__main__':
    main()
