"""Quick sanity check: GPU vs TreeCorr exact on a small random box."""
import sys, time
import numpy as np
import treecorr
sys.path.insert(0, '.')
import gpucorr

rng = np.random.default_rng(1)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
metric = sys.argv[2] if len(sys.argv) > 2 else 'Periodic'
L = 1000.
x, y, z = rng.uniform(0, L, (3, N))
cat = treecorr.Catalog(x=x, y=y, z=z)
kw = dict(min_sep=50., max_sep=150., nbins=50)
if metric == 'Periodic':
    kw.update(period=L)
kw3 = dict(kw, nphi_bins=50, min_phi=0., max_phi=np.pi, bin_type='LogSAS')

for name, cls_cpu, cls_gpu, k in [('NN', treecorr.NNCorrelation, gpucorr.GPUNNCorrelation, kw),
                                  ('NNN', treecorr.NNNCorrelation, gpucorr.GPUNNNCorrelation, kw3)]:
    for prec in ['mixed', 'double']:
        c = cls_cpu(bin_slop=0, **k)
        t = time.perf_counter()
        if name == 'NN': c.process(cat, metric=metric)
        else: c.process(cat, metric=metric, algo='triangle')
        tc = time.perf_counter() - t
        g = cls_gpu(gpu_opts=gpucorr.GPUOptions(gpu_precision=prec), **k)
        t = time.perf_counter()
        g.process(cat, metric=metric)
        tg = time.perf_counter() - t
        if name == 'NN':
            print(f"{name} {prec:6s}: max|dnpairs|={np.max(np.abs(c.npairs-g.npairs)):.3g} "
                  f"tot={c.npairs.sum():.0f} max rel dmeanr={np.max(np.abs(c.meanr-g.meanr)/c.meanr):.2e} "
                  f"dmeanlogr={np.max(np.abs(c.meanlogr-g.meanlogr)):.2e}  cpu {tc:.2f}s gpu {tg:.2f}s")
        else:
            d = c.ntri - g.ntri
            sym = lambda a: a + a.transpose(1, 0, 2)
            ds = sym(c.ntri) - sym(g.ntri)
            m = c.weight > 0
            rel = lambda a, b: np.max(np.abs(a[m] - b[m]) / np.abs(a[m]))
            print(f"{name} {prec:6s}: tot ntri cpu={c.ntri.sum():.0f} gpu={g.ntri.sum():.0f}  "
                  f"bins differing={np.sum(d!=0)} sum|d|={np.abs(d).sum():.0f}  sym bins differing={np.sum(ds!=0)}  "
                  f"marg-phi diff={np.abs(c.ntri.sum((0,1))-g.ntri.sum((0,1))).max():.0f}  "
                  f"exact-path={g.gpu_stats.nexact}  cpu {tc:.2f}s gpu {tg:.2f}s")
            mm = m & (d == 0)
            for q in ['meand1', 'meanlogd1', 'meand2', 'meanlogd2', 'meand3', 'meanlogd3', 'meanphi']:
                a, b = getattr(c, q), getattr(g, q)
                print(f"      {q:10s} max rel diff (bins w/ equal counts) = {np.max(np.abs(a[mm]-b[mm])/np.maximum(np.abs(a[mm]),1e-300)):.2e}")
