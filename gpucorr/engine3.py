"""3-point (LogSAS) exact triplet counting driver."""
import math
import time

import numpy as np

from .engine import (cp, get_module, kern, bstr, ptr, NQ, TRIP_DT, LISTOUT_DT, LISTIN_DT,
                     TRIOUT_DT)


def _pow2_ceil(x):
    return 2.0 ** math.ceil(math.log2(max(x, 1.e-300)))


class TriSetup:
    """Binning constants, reference tables and the global fp64 / uint64 accumulators."""

    def __init__(self, minsep, maxsep, nbins, binsize, minphi, maxphi, nphi, phibinsize,
                 wbound, opts):
        self.nbins, self.nphi = nbins, nphi
        self.ntot = nbins * nbins * nphi
        logminsep = math.log(minsep)
        P = np.zeros((), dtype=TRIP_DT)
        P['minsep'], P['maxsep'], P['logminsep'], P['binsize'] = minsep, maxsep, logminsep, binsize
        P['minphi'], P['maxphi'], P['phibinsize'] = minphi, maxphi, phibinsize
        P['nbins'], P['nphi'] = nbins, nphi
        P['guard'] = opts.guard
        # |www * (value - reference)| bounds for the 64-bit fixed-point sums
        # (d1, logd1, d2, logd2, d3, logd3, phi).  Larger terms go straight to fp64 global.
        wd = maxsep * (1. - math.exp(-binsize))            # widest d bin
        bounds = np.array([
            2. * (2. * wd + maxsep * phibinsize) + 1.,     # d1 - d1(bin centre)
            8.,                                            # log d1 - log d1(bin centre)
            wd, binsize, wd, binsize,                      # d2, log d2, d3, log d3 offsets
            phibinsize]) * 1.01 * max(wbound, 1.e-300)     # phi - phi(bin centre)
        bounds = np.array([_pow2_ceil(b) for b in bounds])
        P['qbound'] = bounds
        P['qscale'] = 2.0 ** 42 / bounds
        P['qinv'] = bounds / 2.0 ** 42
        self.P = P
        self.minsep, self.maxsep, self.logminsep, self.binsize = minsep, maxsep, logminsep, binsize
        k = np.arange(nbins)
        self.refd = cp.asarray(minsep * np.exp(k * binsize))
        self.reflogd = cp.asarray(logminsep + k * binsize)
        rc = minsep * np.exp((k + 0.5) * binsize)
        phic = minphi + (np.arange(nphi) + 0.5) * phibinsize
        d1c = np.sqrt(np.maximum(
            rc[:, None, None] ** 2 + rc[None, :, None] ** 2
            - 2 * rc[:, None, None] * rc[None, :, None] * np.cos(phic)[None, None, :], 0.))
        d1c = np.maximum(d1c, 1.e-6 * minsep)
        self.refd1 = cp.asarray(d1c.ravel())
        self.reflogd1 = cp.asarray(np.log(d1c).ravel())
        self.ntri = cp.zeros(self.ntot, dtype=cp.uint64)
        self.weight = cp.zeros(self.ntot, dtype=cp.float64)
        self.s = [cp.zeros(self.ntot, dtype=cp.float64) for _ in range(NQ)]
        self.nexact = cp.zeros(1, dtype=cp.uint64)
        O = np.zeros((), dtype=TRIOUT_DT)
        O['ntri'] = ptr(self.ntri)
        O['weight'] = ptr(self.weight)
        O['s'] = [ptr(a) for a in self.s]
        O['nexact'] = ptr(self.nexact)
        self.O = O
        same = [(a, b) for a in range(nbins) for b in range(a, nbins)]
        full = [(a, b) for a in range(nbins) for b in range(nbins)]
        self.bp_same = cp.asarray(np.array(same, dtype=np.int32).ravel())
        self.bp_full = cp.asarray(np.array(full, dtype=np.int32).ravel())
        self.n_same, self.n_full = len(same), len(full)

    def results(self):
        cp.cuda.Stream.null.synchronize()
        shape = (self.nbins, self.nbins, self.nphi)
        return (self.ntri.get().astype(np.float64).reshape(shape),
                self.weight.get().reshape(shape),
                [a.get().reshape(shape) for a in self.s],
                int(self.nexact.get()[0]))


class _Timer:
    """Adaptive launch sizing: aim at ~target seconds per kernel launch (keeps each launch
    well below the Windows/WDDM display watchdog, and keeps progress reporting smooth)."""

    def __init__(self, rate0, target=0.25):
        self.target = target
        self.rate = rate0

    def size(self):
        return max(1., self.rate * self.target)

    def update(self, work, dt):
        if dt > 1.e-3 and work > 0:
            self.rate = 0.5 * self.rate + 0.5 * (work / dt)


_timers = {}


def _timer(key, rate0):
    if key not in _timers:
        _timers[key] = _Timer(rate0)
    return _timers[key]


def _ranges(cum, limit):
    """Split items with inclusive cumulative cost `cum` into consecutive ranges of cost
    <= limit (at least one item per range)."""
    out = []
    n = len(cum)
    start = 0
    while start < n:
        base = cum[start - 1] if start else 0.
        end = int(np.searchsorted(cum, base + limit, side='right'))
        end = min(max(end, start + 1), n)
        out.append((start, end))
        start = end
    return out


def _cand_per_center(X):
    g = X.geom
    st = 1
    for n in g.n:
        st *= min(n, 2 * int(g.struct['m']) + 1)
    return max(1., X.n / g.ncell * st)


def _nbr_pass(mod, A, X, excl, c0, nc, setup, opts, counts=None, offsets=None, cursor=None,
              lists=None, dbl=False):
    """Neighbour kernel (pass 0 = count if `counts` given, else pass 1 = fill) on centres
    [c0, c0+nc) of A."""
    nb = setup.nbins
    pss = 0 if counts is not None else 1
    k = kern(mod, 'nbr_kernel<%s,%s,%d,%s>' % (bstr(X.geom.periodic), bstr(excl), pss, bstr(dbl)))
    blk = 128
    cand = _cand_per_center(X)
    tm = _timer(('nbr', opts.device), 2.e9)
    L = lists if lists is not None else np.zeros((), dtype=LISTOUT_DT)
    dummy_i = cp.zeros(1, dtype=cp.int32)
    dummy_l = cp.zeros(1, dtype=cp.int64)
    s = 0
    while s < nc:
        step = int(min(nc - s, max(256, tm.size() / cand)))
        t0 = time.perf_counter()
        args = (A.x, A.y, A.z, np.int32(c0 + s), np.int32(step),
                X.x, X.y, X.z, X.cell_start, X.geom.struct,
                np.float64(setup.minsep), np.float64(setup.maxsep), np.float64(setup.logminsep),
                np.float64(setup.binsize), np.int32(nb), setup.refd, setup.reflogd,
                counts[s * nb:] if counts is not None else dummy_i,
                offsets[s * nb:] if offsets is not None else dummy_l,
                cursor[s * nb:] if cursor is not None else dummy_i,
                L)
        k(((step + blk - 1) // blk,), (blk,), args)
        cp.cuda.Stream.null.synchronize()
        tm.update(step * cand, time.perf_counter() - t0)
        s += step


def nnn_task(A, Bc, Cc, exclB, exclC, setup, opts, stats, progress=None):
    """Accumulate all triangles with vertex P1 in A and the other two vertices either an
    unordered pair from Bc (Cc is None) or one from Bc and one from Cc.  For each triangle the
    2<->3 labelling follows TreeCorr's CCW rule, and P1 is always the centre."""
    mod = get_module(opts.device)
    nb = setup.nbins
    dbl = opts.precision == 'double'
    same = Cc is None
    lists = [(Bc, exclB)] if same else [(Bc, exclB), (Cc, exclC)]
    entry_bytes = 4 if dbl else 24
    unitw = A.unitw and Bc.unitw and (same or Cc.unitw)
    tri = kern(mod, 'tri_kernel<%s,%s,%s,%s>' % (bstr(Bc.geom.periodic), bstr(same), bstr(dbl),
                                                 bstr(unitw)))
    bp = setup.bp_same if same else setup.bp_full
    nbp = setup.n_same if same else setup.n_full
    nsm = cp.cuda.Device(opts.device).attributes['MultiProcessorCount']
    shmem = (NQ * 8 + 8 + 4) * 2 * setup.nphi + 8 * setup.nphi + 16 + 2 * 256 * (16 + 4)
    ttm = _timer(('tri', opts.device, dbl, unitw), 2.e9 if not dbl else 2.e8)
    pool = cp.get_default_memory_pool()

    budget = opts.budget_bytes()
    per_center_counts = nb * (4 + 8 + 4) * len(lists) + 32
    S = int(max(1, min(A.n, (budget // 4) // per_center_counts)))
    for s0 in range(0, A.n, S):
        ns = min(S, A.n - s0)
        cnts = []
        for X, ex in lists:
            c = cp.zeros(ns * nb, dtype=cp.int32)
            _t = time.perf_counter()
            _nbr_pass(mod, A, X, ex, s0, ns, setup, opts, counts=c, dbl=dbl)
            stats.t_nbr += time.perf_counter() - _t
            cnts.append(c)
        offs = []
        for c in cnts:
            o = cp.zeros(ns * nb + 1, dtype=cp.int64)
            cp.cumsum(c, out=o[1:])
            offs.append(o)
        n2 = [c.reshape(ns, nb).astype(cp.float64) for c in cnts]
        if same:
            rs = n2[0].sum(axis=1)
            work = 0.5 * (rs * rs - rs)
            ent = rs
        else:
            work = n2[0].sum(axis=1) * n2[1].sum(axis=1)
            ent = n2[0].sum(axis=1) + n2[1].sum(axis=1)
        work_h = work.get()
        ent_h = ent.get()
        del n2, work, ent, cnts
        stats.ntrip += float(work_h.sum())
        budget = opts.budget_bytes()
        cum_bytes = np.cumsum(ent_h * entry_bytes + nb * 4 * len(lists) + 1.)
        for b0, b1 in _ranges(cum_bytes, budget):
            nc = b1 - b0
            stats.batches += 1
            Lins = []
            keep = []
            for (X, ex), o in zip(lists, offs):
                off = o[b0 * nb: b1 * nb + 1] - o[b0 * nb]
                tot = int(off[-1].get())
                cur = cp.zeros(nc * nb, dtype=cp.int32)
                m = max(tot, 1)
                idx = cp.empty(m, dtype=cp.int32)
                if dbl:
                    arrs = [None] * 5
                else:
                    # vx -> float4 (vx, vy, vz, d - ref) ; lroff -> float
                    arrs = [cp.empty(4 * m, dtype=cp.float32), None, None, None,
                            cp.empty(m, dtype=cp.float32)]
                LO = np.zeros((), dtype=LISTOUT_DT)
                LO['idx'] = ptr(idx)
                for f, a in zip(('vx', 'vy', 'vz', 'roff', 'lroff'), arrs):
                    LO[f] = ptr(a)
                _t = time.perf_counter()
                _nbr_pass(mod, A, X, ex, s0 + b0, nc, setup, opts, offsets=off, cursor=cur,
                          lists=LO, dbl=dbl)
                stats.t_nbr += time.perf_counter() - _t
                del cur
                LI = np.zeros((), dtype=LISTIN_DT)
                for f in ('idx', 'vx', 'vy', 'vz', 'roff', 'lroff'):
                    LI[f] = LO[f]
                LI['px'], LI['py'], LI['pz'], LI['pw'] = ptr(X.x), ptr(X.y), ptr(X.z), ptr(X.w)
                Lins.append((LI, off))
                keep.append((idx, arrs))
                del idx, arrs, off, LO    # only `keep` / `Lins` may hold this batch's buffers
            stats.note_mem()
            LIB = Lins[0]
            LIC = Lins[0] if same else Lins[1]
            cumw = np.cumsum(work_h[s0 - s0 + b0: b1] + 1.)
            t = 0
            while t < nc:
                base = cumw[t - 1] if t else 0.
                t1 = int(np.searchsorted(cumw, base + ttm.size(), side='right'))
                t1 = min(max(t1, t + 1), nc)
                ncl = t1 - t
                wl = float(cumw[t1 - 1] - base)
                avg = wl / (ncl * nbp)
                chunk = int(min(ncl, max(1, 2.e5 / max(avg, 1.e-9))))
                while chunk > 1 and nbp * ((ncl + chunk - 1) // chunk) < 8 * nsm:
                    chunk = max(1, chunk // 2)
                if (ncl + chunk - 1) // chunk > 65535:
                    chunk = (ncl + 65534) // 65535
                nchunk = (ncl + chunk - 1) // chunk
                LIb = LIB[0].copy()
                LIb['offsets'] = ptr(LIB[1]) + 8 * t * nb
                LIc = LIC[0].copy()
                LIc['offsets'] = ptr(LIC[1]) + 8 * t * nb
                t0 = time.perf_counter()
                tri((nbp, nchunk), (opts.block,),
                    (bp, A.x, A.y, A.z, A.w, np.int32(s0 + b0 + t), np.int32(ncl),
                     np.int32(chunk), LIb, LIc, Bc.geom.struct, setup.P, setup.refd,
                     setup.reflogd, setup.refd1, setup.reflogd1, setup.O),
                    shared_mem=shmem)
                cp.cuda.Stream.null.synchronize()
                ttm.update(wl, time.perf_counter() - t0)
                stats.t_tri += time.perf_counter() - t0
                if progress is not None:
                    progress(wl)
                t = t1
            del Lins, LIB, LIC, keep
            pool.free_all_blocks()
        del offs
        pool.free_all_blocks()
