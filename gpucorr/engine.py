"""Low-level CUDA engine: kernel compilation, periodic/Euclidean cell lists, VRAM-adaptive
batching, and the exact pair / triplet counting drivers.

All counting is exact (no bin_slop, no multipole truncation).  See kernels.cu for the formulas,
which are copied from TreeCorr's C++ leaf-level code.
"""
import math
import os
import time

import numpy as np

try:
    import cupy as cp
except ImportError as e:  # pragma: no cover
    cp = None
    _import_error = e

_KERNEL_FILE = os.path.join(os.path.dirname(__file__), 'kernels.cu')
_modules = {}

B = ('false', 'true')
LEG_LM = (16, 32, 64)     # register-array sizes for the Legendre kernel (lmax+1 <= LM)


def require_cupy():
    if cp is None:
        raise ImportError(
            "backend 'cuda' needs CuPy and an NVIDIA GPU.  Install e.g. `pip install cupy-cuda12x` "
            "(CUDA 12, e.g. Google Colab) or `pip install cupy-cuda13x`.  Original error: %r"
            % (_import_error,))
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as e:
        raise RuntimeError("backend 'cuda': no usable CUDA device found (%r)" % (e,))


def _names():
    names = []
    for p in B:
        for a in B:
            names.append('nn_kernel<%s,%s>' % (p, a))
    for p in B:
        for e in B:
            for ps in ('0', '1'):
                for d in B:
                    names.append('nbr_kernel<%s,%s,%s,%s>' % (p, e, ps, d))
    for p in B:
        names.append('cell_kernel<%s>' % p)
        for s in B:
            for d in B:
                for u in B:
                    names.append('tri_kernel<%s,%s,%s,%s>' % (p, s, d, u))
    for p in B:
        for d in B:
            for u in B:
                for lm in LEG_LM:
                    names.append('alm_kernel<%s,%s,%s,%d>' % (p, d, u, lm))
    return names


def get_module(device):
    key = int(device)
    if key not in _modules:
        with open(_KERNEL_FILE) as f:
            src = f.read()
        with cp.cuda.Device(key):
            mod = cp.RawModule(code=src, options=('-std=c++17',), name_expressions=_names())
            mod.compile()
        _modules[key] = mod
    return _modules[key]


def kern(mod, name):
    return mod.get_function(name)


def bstr(x):
    return 'true' if x else 'false'


# --------------------------------------------------------------------------------------------
# Struct layouts (must match kernels.cu; numpy aligned dtypes reproduce the C layout)
# --------------------------------------------------------------------------------------------
GEOM_DT = np.dtype([('Lx', 'f8'), ('Ly', 'f8'), ('Lz', 'f8'),
                    ('x0', 'f8'), ('y0', 'f8'), ('z0', 'f8'),
                    ('cell', 'f8'), ('cx', 'f8'), ('cy', 'f8'), ('cz', 'f8'),
                    ('nx', 'i4'), ('ny', 'i4'), ('nz', 'i4'), ('m', 'i4')], align=True)

NQ = 7
TRIP_DT = np.dtype([('minsep', 'f8'), ('maxsep', 'f8'), ('logminsep', 'f8'), ('binsize', 'f8'),
                    ('minphi', 'f8'), ('maxphi', 'f8'), ('phibinsize', 'f8'),
                    ('nbins', 'i4'), ('nphi', 'i4'),
                    ('guard', 'f8'),
                    ('qscale', 'f8', (NQ,)), ('qinv', 'f8', (NQ,)), ('qbound', 'f8', (NQ,))],
                   align=True)

LISTOUT_DT = np.dtype([(n, 'u8') for n in
                       ('idx', 'vx', 'vy', 'vz', 'roff', 'lroff', 'dvx', 'dvy', 'dvz', 'dd',
                        'dlogd')], align=True)

LISTIN_DT = np.dtype([(n, 'u8') for n in
                      ('idx', 'vx', 'vy', 'vz', 'roff', 'lroff', 'dvx', 'dvy', 'dvz', 'dd',
                       'dlogd', 'offsets', 'px', 'py', 'pz', 'pw')], align=True)

TRIOUT_DT = np.dtype([('ntri', 'u8'), ('weight', 'u8'), ('s', 'u8', (NQ,)), ('nexact', 'u8')],
                     align=True)


def ptr(a):
    return 0 if a is None else int(a.data.ptr)


# --------------------------------------------------------------------------------------------
# Options / memory
# --------------------------------------------------------------------------------------------
class GPUOptions:
    """Options for the cuda backend (all optional)."""

    def __init__(self, gpu_device=0, max_vram_gb=None, gpu_precision='mixed', gpu_guard=1.e-5,
                 gpu_block=256, gpu_method='exact', gpu_lmax=10, verbose=1):
        self.device = int(gpu_device or 0)
        self.max_vram_gb = None if max_vram_gb in (None, '', 0) else float(max_vram_gb)
        if gpu_precision not in ('mixed', 'double'):
            raise ValueError("gpu_precision must be 'mixed' or 'double'")
        self.precision = gpu_precision
        self.guard = float(gpu_guard)
        self.block = int(gpu_block)
        gpu_method = str(gpu_method).lower()
        if gpu_method not in ('exact', 'legendre'):
            raise ValueError("gpu_method must be 'exact' or 'legendre'")
        self.method = gpu_method
        self.lmax = int(gpu_lmax)
        self.verbose = verbose

    @classmethod
    def from_config(cls, cfg):
        keys = ('gpu_device', 'max_vram_gb', 'gpu_precision', 'gpu_guard', 'gpu_block',
                'gpu_method', 'gpu_lmax')
        kw = {k: cfg[k] for k in keys if k in cfg and cfg[k] is not None}
        return cls(**kw)

    def budget_bytes(self):
        """Bytes we may allocate for transient (batched) buffers."""
        pool = cp.get_default_memory_pool()
        pool.free_all_blocks()
        free, total = cp.cuda.runtime.memGetInfo()
        # Under Windows/WDDM memGetInfo may report more than the physical VRAM (the driver
        # can page to system RAM, which is catastrophically slow).  Never plan beyond the
        # physical memory minus a reserve for the driver / display / other processes.
        phys = cp.cuda.Device(self.device).mem_info[1]
        reserve = max(768 * 1024**2, int(0.12 * phys))
        avail = min(free, phys - reserve - pool.used_bytes())
        if self.max_vram_gb is not None:
            cap = self.max_vram_gb * 1024**3 - pool.used_bytes()
            avail = min(avail, cap)
        return max(int(0.5 * avail), 8 * 1024**2)


class Stats:
    def __init__(self):
        self.batches = 0
        self.nexact = 0
        self.ntrip = 0
        self.peak_bytes = 0
        self.time = 0.
        self.t_nbr = 0.
        self.t_tri = 0.

    def note_mem(self):
        pool = cp.get_default_memory_pool()
        self.peak_bytes = max(self.peak_bytes, pool.total_bytes())


# --------------------------------------------------------------------------------------------
# Catalog on device + cell list
# --------------------------------------------------------------------------------------------
class DeviceCat:
    """Positions/weights of a catalog on the GPU, sorted into a cell list."""

    def __init__(self, x, y, z, w, geom_params, mod, block):
        self.n = len(x)
        x = cp.asarray(x, dtype=cp.float64)
        y = cp.asarray(y, dtype=cp.float64)
        z = cp.asarray(z, dtype=cp.float64)
        w = cp.asarray(w, dtype=cp.float64)
        self.geom = geom_params
        g = geom_params.struct
        cid = cp.empty(self.n, dtype=cp.int32)
        k = kern(mod, 'cell_kernel<%s>' % bstr(geom_params.periodic))
        k(((self.n + block - 1) // block,), (block,), (x, y, z, np.int32(self.n), g, cid))
        order = cp.argsort(cid)
        self.order = order                       # sorted -> original index
        self.x = cp.ascontiguousarray(x[order])
        self.y = cp.ascontiguousarray(y[order])
        self.z = cp.ascontiguousarray(z[order])
        self.w = cp.ascontiguousarray(w[order])
        scid = cid[order]
        ncell = geom_params.ncell
        self.cell_start = cp.searchsorted(scid, cp.arange(ncell + 1, dtype=cp.int32)).astype(cp.int32)
        self.sumw = float(cp.sum(self.w))
        self.unitw = bool(cp.all(self.w == 1.0))
        self.wmax = float(cp.max(cp.abs(self.w))) if self.n else 0.
        del cid, scid


class GeomParams:
    """Cell-list geometry.  The grid only enumerates candidates; it never changes which pairs
    are counted (each candidate is tested exactly), so it is not an approximation."""

    def __init__(self, periodic, periods, maxsep, xs, m=2):
        self.periodic = periodic
        s = np.zeros((), dtype=GEOM_DT)
        cmin = maxsep / m * (1. + 1.e-9)        # guarantees m*cell > maxsep strictly
        if periodic:
            L = [float(p) for p in periods]
            n = [max(1, int(Li // cmin)) for Li in L]
            c = [Li / ni for Li, ni in zip(L, n)]
            x0 = [0., 0., 0.]
        else:
            L = [0., 0., 0.]
            lo = [min(float(cp.min(a)) for a in comp) for comp in xs]
            hi = [max(float(cp.max(a)) for a in comp) for comp in xs]
            c0 = cmin
            # cap the number of cells (memory) for very sparse / extended catalogs
            while True:
                n = [int((h - l) // c0) + 1 for l, h in zip(lo, hi)]
                if n[0] * n[1] * n[2] <= (1 << 24):
                    break
                c0 *= 1.5
            c = [c0, c0, c0]
            x0 = lo
        s['Lx'], s['Ly'], s['Lz'] = L
        s['x0'], s['y0'], s['z0'] = x0
        s['cell'] = c[0]
        s['cx'], s['cy'], s['cz'] = c
        s['nx'], s['ny'], s['nz'] = n
        s['m'] = m
        self.struct = s
        self.n = n
        self.ncell = n[0] * n[1] * n[2]


# --------------------------------------------------------------------------------------------
# 2-point
# --------------------------------------------------------------------------------------------
def nn_count(A, Bc, auto, minsep, maxsep, nbins, binsize, opts, stats):
    """Exact Log-binned pair counts (A x B, or i<j pairs of A if auto; then A is Bc).
    Returns npairs, weight, sum(w r), sum(w log r) as numpy arrays."""
    mod = get_module(opts.device)
    out = cp.zeros(4 * nbins, dtype=cp.float64)
    k = kern(mod, 'nn_kernel<%s,%s>' % (bstr(Bc.geom.periodic), bstr(auto)))
    blk = opts.block
    step = 1 << 16          # keep each launch short (display watchdog on Windows/WDDM)
    for s0 in range(0, A.n, step):
        n0 = min(step, A.n - s0)
        k(((n0 + blk - 1) // blk,), (blk,),
          (A.x, A.y, A.z, A.w, np.int32(s0), np.int32(n0),
           Bc.x, Bc.y, Bc.z, Bc.w, Bc.cell_start, Bc.geom.struct,
           np.float64(minsep * minsep), np.float64(maxsep * maxsep),
           np.float64(math.log(minsep)), np.float64(binsize), np.int32(nbins), out),
          shared_mem=4 * nbins * 8)
    cp.cuda.Stream.null.synchronize()
    stats.note_mem()
    o = out.get().reshape(4, nbins)
    return o[0], o[1], o[2], o[3]
