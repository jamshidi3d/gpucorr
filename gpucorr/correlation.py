"""TreeCorr-compatible correlation classes whose pair/triplet accumulation runs on the GPU.

Only `process_auto` / `process_cross` (2pt) and `process_auto` / `process_cross12` (3pt) are
replaced.  Everything else -- catalog reading, finalize, calculateXi / calculateZeta, variance,
write -- is TreeCorr's own code, so outputs have exactly TreeCorr's format and estimators.

The GPU counts every pair/triplet exactly (like TreeCorr with bin_slop=0 and, for 3pt,
algo='triangle'); bin_slop / angle_slop / algo / max_n settings are therefore ignored.
"""
import time

import numpy as np
import treecorr

from . import engine
from .engine import GPUOptions, Stats, DeviceCat, GeomParams


def _cat_xyz(cat):
    if cat.coords != '3d':
        raise NotImplementedError(
            "gpucorr currently supports 3d (x,y,z) catalogs; got coords=%r" % cat.coords)
    w = cat.w if cat.w is not None else np.ones(cat.ntot)
    return cat.x, cat.y, cat.z, w


class _GPUMixin:
    """Shared helpers.  `_gpu_opts` is set by the factory / constructor."""

    _gpu_opts = None

    def _gpu_init(self, gpu_opts):
        engine.require_cupy()
        if gpu_opts is None:
            gpu_opts = GPUOptions.from_config(self.config)
        self._gpu_opts = gpu_opts
        self.gpu_stats = Stats()
        self._devcache = {}

    def _metric_info(self):
        if self.metric not in ('Euclidean', 'Periodic'):
            raise NotImplementedError("gpucorr supports metric Euclidean or Periodic, not %r"
                                      % self.metric)
        periodic = self.metric == 'Periodic'
        periods = (self._ro.xperiod, self._ro.yperiod, self._ro.zperiod)
        return periodic, periods

    def _devcat(self, cat, periodic, periods, maxsep):
        key = (id(cat), periodic, periods, maxsep)
        if key not in self._devcache:
            opts = self._gpu_opts
            with engine.cp.cuda.Device(opts.device):
                x, y, z, w = _cat_xyz(cat)
                xd = [engine.cp.asarray(a, dtype=engine.cp.float64) for a in (x, y, z)]
                geom = GeomParams(periodic, periods, maxsep, [[a] for a in xd])
                mod = engine.get_module(opts.device)
                self._devcache[key] = (cat, DeviceCat(xd[0], xd[1], xd[2], w, geom, mod,
                                                      opts.block))
        return self._devcache[key][1]

    def _check_patches(self, *cats):
        for c in cats:
            if c is not None and c.npatch not in (1, None):
                raise NotImplementedError("gpucorr: catalogs with npatch > 1 are not supported "
                                          "yet (use backend: cpu for patch-based covariances).")


class GPUNNCorrelation(_GPUMixin, treecorr.NNCorrelation):
    """`treecorr.NNCorrelation` with exact GPU pair counting (Log binning)."""

    def __init__(self, config=None, *, logger=None, gpu_opts=None, **kwargs):
        if config is not None:
            config = {k: v for k, v in config.items() if not _is_gpu_key(k)}
        super().__init__(config, logger=logger, **kwargs)
        if self.bin_type != 'Log':
            raise NotImplementedError("gpucorr NN supports bin_type='Log' only")
        self._gpu_init(gpu_opts)

    def _run(self, cat1, cat2, auto):
        periodic, periods = self._metric_info()
        ro = self._ro
        opts = self._gpu_opts
        t0 = time.perf_counter()
        with engine.cp.cuda.Device(opts.device):
            A = self._devcat(cat1, periodic, periods, ro.max_sep)
            Bc = A if auto else self._devcat(cat2, periodic, periods, ro.max_sep)
            npairs, weight, sr, slr = engine.nn_count(A, Bc, auto, ro.min_sep, ro.max_sep,
                                                      ro.nbins, ro.bin_size, opts,
                                                      self.gpu_stats)
        self.gpu_stats.time += time.perf_counter() - t0
        self.npairs += npairs
        self.weight += weight
        if not self._corr_only:
            self.meanr += sr
            self.meanlogr += slr

    def process_auto(self, cat, *, metric=None, num_threads=None, corr_only=False):
        self._check_patches(cat)
        self._set_metric(metric, cat.coords)
        self._corr_only = corr_only
        self._meanww = cat.meanw ** 2
        self.logger.info('GPU: NN auto-correlation, N=%d', cat.ntot)
        self._run(cat, None, True)
        self.tot += 0.5 * cat.sumw ** 2

    def process_cross(self, cat1, cat2, *, metric=None, num_threads=None, corr_only=False):
        self._check_patches(cat1, cat2)
        self._set_metric(metric, cat1.coords, cat2.coords)
        self._corr_only = corr_only
        self._meanww = cat1.meanw * cat2.meanw
        self.logger.info('GPU: NN cross-correlation, N1=%d, N2=%d', cat1.ntot, cat2.ntot)
        self._run(cat1, cat2, False)
        self.tot += cat1.sumw * cat2.sumw


class GPUNNNCorrelation(_GPUMixin, treecorr.NNNCorrelation):
    """`treecorr.NNNCorrelation` (bin_type='LogSAS') with exact GPU triangle counting."""

    def __init__(self, config=None, *, logger=None, gpu_opts=None, **kwargs):
        if config is not None:
            config = {k: v for k, v in config.items() if not _is_gpu_key(k)}
        super().__init__(config, logger=logger, **kwargs)
        if self.bin_type != 'LogSAS':
            raise NotImplementedError("gpucorr NNN supports bin_type='LogSAS' only")
        self._gpu_init(gpu_opts)

    def process(self, cat1, cat2=None, cat3=None, **kwargs):
        # The GPU path is exact: never take TreeCorr's multipole branch.
        kwargs['algo'] = 'triangle'
        kwargs.pop('max_n', None)
        return super().process(cat1, cat2, cat3, **kwargs)

    def _setup(self, wbound):
        ro = self._ro
        return engine3.TriSetup(ro.min_sep, ro.max_sep, ro.nbins, ro.bin_size,
                                ro.min_u, ro.max_u, ro.nubins, ro.ubin_size, wbound,
                                self._gpu_opts)

    def _accumulate(self, setup):
        ntri, weight, s, nexact = setup.results()
        self.ntri += ntri
        self.weightr += weight
        if not self._corr_only:
            self.meand1 += s[0]
            self.meanlogd1 += s[1]
            self.meand2 += s[2]
            self.meanlogd2 += s[3]
            self.meand3 += s[4]
            self.meanlogd3 += s[5]
            self.meanu += s[6]
        self.gpu_stats.nexact += nexact

    def _tasks(self, tasks, wbound):
        opts = self._gpu_opts
        t0 = time.perf_counter()
        with engine.cp.cuda.Device(opts.device):
            if opts.method == 'legendre':
                ro = self._ro
                setup = engine_legendre.LegSetup(ro.min_sep, ro.max_sep, ro.nbins, ro.bin_size,
                                                 opts.lmax, opts)
                for (A, Bc, Cc, eB, eC) in tasks:
                    engine_legendre.nnn_legendre_task(A, Bc, Cc, eB, eC, setup, opts,
                                                      self.gpu_stats)
                self._accumulate_legendre(setup)
            else:
                setup = self._setup(wbound)
                for (A, Bc, Cc, eB, eC) in tasks:
                    engine3.nnn_task(A, Bc, Cc, eB, eC, setup, opts, self.gpu_stats)
                self._accumulate(setup)
            del setup
            engine.cp.get_default_memory_pool().free_all_blocks()
        self.gpu_stats.time += time.perf_counter() - t0

    def _accumulate_legendre(self, setup):
        """Store the multipoles and fill TreeCorr's (d2, d3, phi) arrays from them.

        legendre[l, k2, k3] = sum over triangles (TreeCorr counting: each triangle once per
        vertex as P1, d2<->d3 symmetrised and halved) of w * P_l(cos phi).  The phi-binned
        weights are the truncated Legendre series integrated over each phi bin; d2/d3 means
        are exact per (k2, k3); phi / d1 means are bin-centre values."""
        M, W, D2, L2 = setup.results()
        ro = self._ro
        lmax = setup.lmax
        half = 0.5 * M                                          # ordered -> TreeCorr convention
        half[0] = 0.5 * W            # monopole from the exact moment sums (no fp32 Y_00 rounding)
        if not hasattr(self, 'legendre') or self.legendre is None or self.legendre.shape != half.shape:
            self.legendre = np.zeros_like(half)
        self.legendre += half
        J = engine_legendre.phi_bin_projector(lmax, ro.min_u, ro.max_u, ro.nubins)
        wb = np.einsum('lab,lk->abk', half, J)                  # (k2, k3, kphi)
        self.weightr += wb
        www = self._meanwww if getattr(self, '_meanwww', 0) else 1.
        self.ntri += wb / www
        if not self._corr_only:
            with np.errstate(divide='ignore', invalid='ignore'):
                md2 = np.where(W != 0, D2 / W, 0.)
                ml2 = np.where(W != 0, L2 / W, 0.)
            md3, ml3 = md2.T, ml2.T
            phic = ro.min_u + (np.arange(ro.nubins) + 0.5) * ro.ubin_size
            md1 = np.sqrt(np.maximum(md2[:, :, None] ** 2 + md3[:, :, None] ** 2
                                     - 2 * md2[:, :, None] * md3[:, :, None] * np.cos(phic), 1e-300))
            self.meand2 += wb * md2[:, :, None]
            self.meanlogd2 += wb * ml2[:, :, None]
            self.meand3 += wb * md3[:, :, None]
            self.meanlogd3 += wb * ml3[:, :, None]
            self.meanu += wb * phic[None, None, :]
            self.meand1 += wb * md1
            self.meanlogd1 += wb * np.log(md1)

    def write(self, file_name, **kwargs):
        super().write(file_name, **kwargs)
        if self._gpu_opts.method == 'legendre' and getattr(self, 'legendre', None) is not None:
            write_legendre(self, file_name, rrr=kwargs.get('rrr'), drr=kwargs.get('drr'),
                           rdd=kwargs.get('rdd'))

    def process_auto(self, cat, *, metric=None, num_threads=None, corr_only=False):
        self._check_patches(cat)
        self._set_metric(metric, cat.coords)
        self._corr_only = corr_only
        self._meanwww = cat.meanw ** 3
        periodic, periods = self._metric_info()
        self.logger.info('GPU: NNN auto-correlation, N=%d', cat.ntot)
        with engine.cp.cuda.Device(self._gpu_opts.device):
            A = self._devcat(cat, periodic, periods, self._ro.max_sep)
        # every triangle once per vertex as P1 (TreeCorr's three rotations)
        self._tasks([(A, A, None, True, False)], A.wmax ** 3)
        self.tot += (1. / 6.) * cat.sumw ** 3

    def process_cross12(self, cat1, cat2, *, metric=None, ordered=True, num_threads=None,
                        corr_only=False):
        self._check_patches(cat1, cat2)
        self._set_metric(metric, cat1.coords, cat2.coords)
        self._corr_only = corr_only
        self._meanwww = cat1.meanw * cat2.meanw ** 2
        periodic, periods = self._metric_info()
        self.logger.info('GPU: NNN cross12 (ordered=%s), N1=%d, N2=%d', ordered, cat1.ntot,
                         cat2.ntot)
        ms = self._ro.max_sep
        with engine.cp.cuda.Device(self._gpu_opts.device):
            A = self._devcat(cat1, periodic, periods, ms)
            Bc = self._devcat(cat2, periodic, periods, ms)
        tasks = [(A, Bc, None, False, False)]       # P1 from cat1, {P2,P3} from cat2
        if not ordered:
            # remaining rotations: P1 from cat2, the other two = one cat1 + one other cat2
            tasks.append((Bc, A, Bc, False, True))
        self._tasks(tasks, A.wmax * Bc.wmax ** 2)
        self.tot += 0.5 * cat1.sumw * cat2.sumw ** 2

    def process_cross21(self, *args, **kwargs):
        raise NotImplementedError("gpucorr: process_cross21 is not implemented (NN-N only).")

    def process_cross(self, *args, **kwargs):
        raise NotImplementedError("gpucorr: three-catalog NNN cross-correlations are not "
                                  "implemented yet; use backend: cpu.")


from . import engine3  # noqa: E402  (after class definitions to avoid import cycles)
from . import engine_legendre  # noqa: E402

GPU_KEYS = ('backend', 'gpu_device', 'max_vram_gb', 'gpu_precision', 'gpu_guard', 'gpu_block',
            'gpu_method', 'gpu_lmax')


def legendre_zeta(ddd, rrr=None, drr=None, rdd=None):
    """Multipoles of the full estimator.  Returns (zeta_l, Z_l) with
    Z_l = DDD_l - RDD_l f + DRR_l f - RRR_l f  and  zeta_l = Z_l / (RRR_0 f)   (the RRR
    normalisation uses only its monopole, exact for an isotropic RRR such as a periodic box),
    so that  zeta(cos phi) ~= sum_l (2l+1) zeta_l P_l(cos phi)."""
    Z = ddd.legendre.copy()
    if rrr is None:
        return None, Z
    rrrf = ddd.tot / rrr.tot
    if drr is not None and rdd is not None:
        Z = Z - rdd.legendre * (ddd.tot / rdd.tot) + drr.legendre * (ddd.tot / drr.tot)
    Z = Z - rrr.legendre * rrrf
    den = rrr.legendre[0] * rrrf
    with np.errstate(divide='ignore', invalid='ignore'):
        zeta = np.where(den != 0, Z / den[None], 0.)
    return zeta, Z


def write_legendre(ddd, file_name, rrr=None, drr=None, rdd=None):
    import os
    root, ext = os.path.splitext(file_name)
    out = root + '_legendre' + (ext or '.txt')
    zeta, Z = legendre_zeta(ddd, rrr, drr, rdd)
    ro = ddd._ro
    lmax = ddd.legendre.shape[0] - 1
    rn = ro.min_sep * np.exp((np.arange(ro.nbins) + 0.5) * ro.bin_size)
    cols = ['d2_nom', 'd3_nom', 'l', 'DDD_l']
    rows = []
    for a in range(ro.nbins):
        for b in range(ro.nbins):
            for l in range(lmax + 1):
                r = [rn[a], rn[b], l, ddd.legendre[l, a, b]]
                if rrr is not None:
                    r += [rrr.legendre[l, a, b]]
                    if drr is not None and rdd is not None:
                        r += [drr.legendre[l, a, b], rdd.legendre[l, a, b]]
                    r += [Z[l, a, b], zeta[l, a, b]]
                rows.append(r)
    if rrr is not None:
        cols += ['RRR_l'] + (['DRR_l', 'RDD_l'] if drr is not None and rdd is not None else []) + \
            ['Z_l', 'zeta_l']
    hdr = ('Legendre multipoles (Slepian & Eisenstein) of the NNN counts, lmax=%d. X_l = sum w P_l(cos phi) '
           'per (d2,d3) bin in TreeCorr counting; Z_l = DDD_l - RDD_l*f + DRR_l*f - RRR_l*f (TreeCorr tot '
           'factors); zeta_l = Z_l / (RRR_0 f): zeta(phi) ~ sum_l (2l+1) zeta_l P_l(cos phi).\n' % lmax
           + ' '.join('%12s' % c for c in cols))
    np.savetxt(out, np.array(rows), fmt='%.6e', header=hdr)
    ddd.logger.warning('Wrote Legendre multipoles to %s', out)


def _is_gpu_key(k):
    return k in GPU_KEYS
