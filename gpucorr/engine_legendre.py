"""Legendre / spherical-harmonic 3PCF mode (Slepian & Eisenstein 2015) on the GPU.

For every centre c and radial bin k:   a_lm(c,k) = sum_{j in bin k} w_j Y_lm(r_hat_cj)
By the addition theorem, for ordered neighbour pairs (j in k2, l in k3):

    M_l(k2,k3) = sum_c w_c sum_{j,l} w_j w_l P_l(cos phi_jl)
               = sum_c w_c 4pi/(2l+1) [a_l0 a_l0* + 2 Re sum_{m>0} a_lm(k2) a_lm(k3)*]   (- self pairs)

which is a complex matrix product over all centres (cuBLAS).  Cost ~ N * n * lmax^2 instead
of N * n^2.  The multipoles are exact (to floating point) for l <= lmax; binned-in-phi counts
are obtained from them by a truncated Legendre series (an approximation that converges as
lmax -> infinity).

Convention: M is accumulated over *ordered* pairs, i.e. it is the Legendre transform of the
d2<->d3-symmetrised exact counts  S(k2,k3) + S(k3,k2).  The values handed to TreeCorr are M/2.
"""
import math
import time

import numpy as np

from .engine import cp, get_module, kern, bstr, ptr, LISTOUT_DT, LISTIN_DT, LEG_LM
from .engine3 import _nbr_pass, _ranges, _timer


def lm_index(l, m):
    return l * (l + 1) // 2 + m


def recursion_coefs(lmax):
    nlm = (lmax + 1) * (lmax + 2) // 2
    ca = np.zeros(nlm)
    cb = np.zeros(nlm)
    for l in range(2, lmax + 1):
        for m in range(0, l - 1):
            i = lm_index(l, m)
            ca[i] = math.sqrt((4. * l * l - 1.) / (l * l - m * m))
            cb[i] = math.sqrt(((l - 1.) ** 2 - m * m) / (4. * (l - 1.) ** 2 - 1.))
    return ca, cb


class LegSetup:
    def __init__(self, minsep, maxsep, nbins, binsize, lmax, opts):
        if lmax < 0 or lmax + 1 > max(LEG_LM):
            raise ValueError("gpu_lmax must be in [0, %d]" % (max(LEG_LM) - 1))
        self.minsep, self.maxsep, self.nbins, self.binsize = minsep, maxsep, nbins, binsize
        self.logminsep = math.log(minsep)
        self.lmax = lmax
        self.nlm = (lmax + 1) * (lmax + 2) // 2
        self.LM = min(x for x in LEG_LM if x >= lmax + 1)
        k = np.arange(nbins)
        self.refd = cp.asarray(minsep * np.exp(k * binsize))
        self.reflogd = cp.asarray(self.logminsep + k * binsize)
        ca, cb = recursion_coefs(lmax)
        self.ca, self.cb = cp.asarray(ca), cp.asarray(cb)
        self.M = cp.zeros((lmax + 1, nbins, nbins))      # ordered-pair Legendre sums
        self.W = cp.zeros((nbins, nbins))                 # sum w      (== M[0])
        self.D2 = cp.zeros((nbins, nbins))                # sum w d    of the member in bin k2
        self.L2 = cp.zeros((nbins, nbins))                # sum w logd of the member in bin k2

    def results(self):
        cp.cuda.Stream.null.synchronize()
        return self.M.get(), self.W.get(), self.D2.get(), self.L2.get()


def _alm_pass(mod, A, X, LI, c0, nc, setup, opts, dbl, unitw, stats):
    """a_lm and moments for all (centre, bin) slots of a filled neighbour list."""
    nb, nlm = setup.nbins, setup.nlm
    nslots = nc * nb
    alm = cp.zeros(nslots * 2 * nlm, dtype=cp.float64)
    mom = cp.zeros(nslots * 6, dtype=cp.float64)
    k = kern(mod, 'alm_kernel<%s,%s,%s,%d>' % (bstr(X.geom.periodic), bstr(dbl), bstr(unitw),
                                              setup.LM))
    tb = 8 if dbl else 4
    L1 = setup.lmax + 1
    G = max(1, min(128 // L1, (49152 // tb - 6) // (2 * nlm)))
    block = L1 * G
    shmem = (G * 2 * nlm + 6) * tb
    tm = _timer(('alm', opts.device, dbl, setup.lmax), 2.e7)
    off = LI[1]
    s = 0
    while s < nslots:
        step = int(min(nslots - s, max(nb, tm.size())))
        step = max(nb, (step // nb) * nb)          # whole centres per launch
        L = LI[0].copy()
        L['offsets'] = ptr(off) + 8 * s
        t0 = time.perf_counter()
        k((step,), (block,),
          (L, A.x, A.y, A.z, np.int32(c0 + s // nb), np.int32(step), np.int32(nb), np.int32(setup.lmax),
           setup.refd, setup.ca, setup.cb, X.geom.struct, alm[s * 2 * nlm:], mom[s * 6:]),
          shared_mem=shmem)
        cp.cuda.Stream.null.synchronize()
        tm.update(step, time.perf_counter() - t0)
        s += step
    return alm.view(cp.complex128).reshape(nc, nb, nlm), mom.reshape(nc, nb, 6)


def _combine(setup, almB, momB, almC, momC, wc, same, dbl=True):
    lmax = setup.lmax
    ct = cp.complex128 if dbl else cp.complex64
    nc, nb, _ = almB.shape
    for l in range(lmax + 1):
        sl = slice(lm_index(l, 0), lm_index(l, l) + 1)
        f = cp.full(l + 1, 2.0)
        f[0] = 1.0
        XB = (almB[:, :, sl] * (wc[:, None, None] * f[None, None, :])).transpose(1, 0, 2).reshape(nb, -1).astype(ct)
        XC = almC[:, :, sl].transpose(1, 0, 2).reshape(nb, -1).astype(ct)
        R = (XB @ XC.conj().T).real.astype(cp.float64) * (4. * math.pi / (2 * l + 1))
        setup.M[l] += R if same else R + R.T
    s0B, s1B, slB = momB[..., 0], momB[..., 1], momB[..., 2]
    s0C, s1C, slC = momC[..., 0], momC[..., 1], momC[..., 2]
    w = wc[:, None]
    if same:
        setup.W += (w * s0B).T @ s0C
        setup.D2 += (w * s1B).T @ s0C
        setup.L2 += (w * slB).T @ s0C
        # remove j == l (ordered self pairs): P_l(1) = 1 for every l
        q0 = (w * momB[..., 3]).sum(0)
        q1 = (w * momB[..., 4]).sum(0)
        ql = (w * momB[..., 5]).sum(0)
        idx = cp.arange(nb)
        setup.M[:, idx, idx] -= q0[None, :]
        setup.W[idx, idx] -= q0
        setup.D2[idx, idx] -= q1
        setup.L2[idx, idx] -= ql
    else:
        setup.W += (w * s0B).T @ s0C + (w * s0C).T @ s0B
        setup.D2 += (w * s1B).T @ s0C + (w * s1C).T @ s0B
        setup.L2 += (w * slB).T @ s0C + (w * slC).T @ s0B


def nnn_legendre_task(A, Bc, Cc, exclB, exclC, setup, opts, stats):
    """Same task decomposition as engine3.nnn_task (P1 in A; the other two vertices an
    unordered pair from Bc, or one from Bc and one from Cc)."""
    mod = get_module(opts.device)
    nb, nlm = setup.nbins, setup.nlm
    dbl = opts.precision == 'double'
    same = Cc is None
    lists = [(Bc, exclB)] if same else [(Bc, exclB), (Cc, exclC)]
    entry_bytes = 4 if dbl else 24
    unitw = A.unitw and Bc.unitw and (same or Cc.unitw)
    pool = cp.get_default_memory_pool()

    # per-centre transient bytes: list entries + a_lm/moments per list + matmul copies
    per_slot = len(lists) * (2 * nlm * 8 + 6 * 8) + 2 * nlm * 16 + 8
    budget = opts.budget_bytes()
    per_center_counts = nb * (4 + 8 + 4) * len(lists) + 32
    S = int(max(1, min(A.n, (budget // 4) // per_center_counts)))
    for s0 in range(0, A.n, S):
        ns = min(S, A.n - s0)
        cnts = []
        for X, ex in lists:
            c = cp.zeros(ns * nb, dtype=cp.int32)
            t_ = time.perf_counter()
            _nbr_pass(mod, A, X, ex, s0, ns, setup, opts, counts=c, dbl=dbl)
            stats.t_nbr += time.perf_counter() - t_
            cnts.append(c)
        offs = []
        for c in cnts:
            o = cp.zeros(ns * nb + 1, dtype=cp.int64)
            cp.cumsum(c, out=o[1:])
            offs.append(o)
        ent = sum(c.reshape(ns, nb).sum(axis=1).astype(cp.float64) for c in cnts).get()
        del cnts
        budget = opts.budget_bytes()
        cum_bytes = np.cumsum(ent * entry_bytes + nb * per_slot + nb * 4 * len(lists) + 1.)
        for b0, b1 in _ranges(cum_bytes, budget):
            nc = b1 - b0
            stats.batches += 1
            res = []
            for (X, ex), o in zip(lists, offs):
                off = o[b0 * nb: b1 * nb + 1] - o[b0 * nb]
                tot = int(off[-1].get())
                cur = cp.zeros(nc * nb, dtype=cp.int32)
                m = max(tot, 1)
                idx = cp.empty(m, dtype=cp.int32)
                arrs = [None] * 5 if dbl else [cp.empty(4 * m, dtype=cp.float32), None, None, None,
                                                cp.empty(m, dtype=cp.float32)]
                LO = np.zeros((), dtype=LISTOUT_DT)
                LO['idx'] = ptr(idx)
                for f, a in zip(('vx', 'vy', 'vz', 'roff', 'lroff'), arrs):
                    LO[f] = ptr(a)
                t_ = time.perf_counter()
                _nbr_pass(mod, A, X, ex, s0 + b0, nc, setup, opts, offsets=off, cursor=cur,
                          lists=LO, dbl=dbl)
                stats.t_nbr += time.perf_counter() - t_
                del cur
                LI = np.zeros((), dtype=LISTIN_DT)
                for f in ('idx', 'vx', 'vy', 'vz', 'roff', 'lroff'):
                    LI[f] = LO[f]
                LI['px'], LI['py'], LI['pz'], LI['pw'] = ptr(X.x), ptr(X.y), ptr(X.z), ptr(X.w)
                t_ = time.perf_counter()
                res.append(_alm_pass(mod, A, X, (LI, off), s0 + b0, nc, setup, opts, dbl, unitw, stats))
                stats.t_tri += time.perf_counter() - t_
                stats.ntrip += float(tot)
                del idx, arrs, off, LO, LI
                pool.free_all_blocks()
            stats.note_mem()
            wc = A.w[s0 + b0: s0 + b1]
            t_ = time.perf_counter()
            almB, momB = res[0]
            almC, momC = res[0] if same else res[1]
            _combine(setup, almB, momB, almC, momC, wc, same, dbl)
            cp.cuda.Stream.null.synchronize()
            stats.t_tri += time.perf_counter() - t_
            stats.t_comb = getattr(stats, 't_comb', 0.) + time.perf_counter() - t_
            del res, almB, momB, almC, momC
            pool.free_all_blocks()
        del offs
        pool.free_all_blocks()


# --------------------------------------------------------------------------------------------
# From multipoles to TreeCorr's phi bins
# --------------------------------------------------------------------------------------------
def legendre_P(lmax, x):
    """P_0..P_lmax at x (numpy), returns (lmax+1, len(x))."""
    x = np.asarray(x, float)
    P = np.zeros((lmax + 2,) + x.shape)
    P[0] = 1.
    if lmax + 1 >= 1:
        P[1] = x
    for l in range(1, lmax + 1):
        P[l + 1] = ((2 * l + 1) * x * P[l] - l * P[l - 1]) / (l + 1)
    return P


def phi_bin_projector(lmax, minphi, maxphi, nphi):
    """J[l, kphi] = (2l+1)/2 * integral of P_l(mu) over the bin's mu range, so that
    counts(kphi) = sum_l M_l J[l, kphi]  for the truncated series n(mu) = sum (2l+1)/2 M_l P_l."""
    edges = minphi + (maxphi - minphi) * np.arange(nphi + 1) / nphi
    mu = np.cos(edges)                     # decreasing
    P = legendre_P(lmax + 1, mu)
    J = np.zeros((lmax + 1, nphi))
    hi, lo = mu[:-1], mu[1:]
    J[0] = 0.5 * (hi - lo)
    for l in range(1, lmax + 1):
        F = (P[l + 1] - P[l - 1])          # (2l+1) * integral
        J[l] = 0.5 * (F[:-1] - F[1:])
    return J
