// gpucorr CUDA kernels: exact (brute-force over a cell list) NN and NNN-LogSAS counting.
//
// Every candidate pair/triplet is tested with the same formulas TreeCorr (v5.1) uses at the
// leaf level (include/Metric.h, include/BinType.h, src/Corr2.cpp, src/Corr3.cpp):
//   * Periodic displacement:  d = p2 - p1, then  while(d > L/2) d -= L; while(d < -L/2) d += L;
//   * 2pt Log bin:  minsep^2 <= rsq < maxsep^2,  k = int((log(sqrt(rsq)) - log(minsep))/binsize)
//   * 3pt LogSAS:   d2 = |P1P3|, d3 = |P1P2|, d1 = |P2P3|;  cos(phi) = (d2^2+d3^2-d1^2)/(2 d2 d3)
//                   2<->3 chosen such that CCW(P1,P3,P2):  cross(r13, r12) . p1 < 0
//
// Compiled at runtime by CuPy/NVRTC.  Template parameters are instantiated from Python via
// name_expressions.

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned long long u64;
template <bool B> struct SelT { typedef float type; };
template <> struct SelT<true> { typedef double type; };
typedef unsigned long long uintptr_t;
typedef long long i64;

#define SQR(x) ((x)*(x))

__device__ __forceinline__ double wrap1(double x, double L)
{
    while (x > L/2.) x -= L;
    while (x < -L/2.) x += L;
    return x;
}

struct Geom {
    double Lx, Ly, Lz;      // periods (unused if !PERIODIC)
    double x0, y0, z0;      // grid origin
    double cell;            // grid cell size (same in all dims for Euclidean); periodic uses L/n
    double cx, cy, cz;      // cell sizes per dim
    int nx, ny, nz;         // number of cells per dim
    int m;                  // stencil half-width in cells
};

template <bool PERIODIC>
__device__ __forceinline__ void disp(double x1, double y1, double z1,
                                     double x2, double y2, double z2,
                                     const Geom& g, double& dx, double& dy, double& dz)
{
    dx = x2 - x1; dy = y2 - y1; dz = z2 - z1;
    if (PERIODIC) { dx = wrap1(dx, g.Lx); dy = wrap1(dy, g.Ly); dz = wrap1(dz, g.Lz); }
}

// cell coordinate of a point along one dim
template <bool PERIODIC>
__device__ __forceinline__ int cellcoord(double x, double x0, double c, int n, double L)
{
    int i;
    if (PERIODIC) {
        double xm = x - L * floor(x / L);   // into [0, L)
        i = int(xm / c);
    } else {
        i = int(floor((x - x0) / c));
    }
    if (PERIODIC) { if (i >= n) i = n-1; if (i < 0) i = 0; }
    return i;
}

// Iterate over the stencil cells along one dim.  Writes the list of (unique) cell indices.
template <bool PERIODIC>
__device__ __forceinline__ int stencil1(int i, int n, int m, int* out)
{
    int cnt = 0;
    if (PERIODIC) {
        if (2*m+1 >= n) { for (int k=0;k<n;++k) out[cnt++] = k; }
        else { for (int d=-m; d<=m; ++d) { int k = i+d; k = ((k % n) + n) % n; out[cnt++] = k; } }
    } else {
        for (int d=-m; d<=m; ++d) { int k = i+d; if (k>=0 && k<n) out[cnt++] = k; }
    }
    return cnt;
}

// Cell index of each point (same cellcoord() as used by the stencil search).
template <bool PERIODIC>
__global__ void cell_kernel(const double* __restrict__ x, const double* __restrict__ y,
                            const double* __restrict__ z, int n, Geom g, int* cid)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ix = cellcoord<PERIODIC>(x[i], g.x0, g.cx, g.nx, g.Lx);
    int iy = cellcoord<PERIODIC>(y[i], g.y0, g.cy, g.ny, g.Ly);
    int iz = cellcoord<PERIODIC>(z[i], g.z0, g.cz, g.nz, g.Lz);
    if (!PERIODIC) {   // grid is built from the bounding box of this catalog: always in range
        ix = max(0, min(g.nx-1, ix)); iy = max(0, min(g.ny-1, iy)); iz = max(0, min(g.nz-1, iz));
    }
    cid[i] = (ix*g.ny + iy)*g.nz + iz;
}

#define MAXST 64

// ---------------------------------------------------------------------------------------
// 2-point NN.  One thread per center i in A.  Neighbours from catalog B (cell-sorted).
// AUTO: A == B (same sorted arrays) and only pairs with j > i are counted.
// out layout: [npairs(nb) | weight(nb) | sumr(nb) | sumlogr(nb)]  (double)
// ---------------------------------------------------------------------------------------
template <bool PERIODIC, bool AUTO>
__global__ void nn_kernel(const double* __restrict__ ax, const double* __restrict__ ay,
                          const double* __restrict__ az, const double* __restrict__ aw,
                          int i0, int na,
                          const double* __restrict__ bx, const double* __restrict__ by,
                          const double* __restrict__ bz, const double* __restrict__ bw,
                          const int* __restrict__ cell_start,
                          Geom g, double minsepsq, double maxsepsq, double logminsep,
                          double binsize, int nbins, double* out)
{
    extern __shared__ double sh[];   // 4*nbins
    for (int t=threadIdx.x; t<4*nbins; t+=blockDim.x) sh[t] = 0.;
    __syncthreads();

    int i = i0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < i0 + na) {
        const double x1 = ax[i], y1 = ay[i], z1 = az[i], w1 = aw[i];
        int sx[MAXST], sy[MAXST], sz[MAXST];
        int nsx = stencil1<PERIODIC>(cellcoord<PERIODIC>(x1, g.x0, g.cx, g.nx, g.Lx), g.nx, g.m, sx);
        int nsy = stencil1<PERIODIC>(cellcoord<PERIODIC>(y1, g.y0, g.cy, g.ny, g.Ly), g.ny, g.m, sy);
        int nsz = stencil1<PERIODIC>(cellcoord<PERIODIC>(z1, g.z0, g.cz, g.nz, g.Lz), g.nz, g.m, sz);
        for (int a=0;a<nsx;++a) for (int b=0;b<nsy;++b) for (int c=0;c<nsz;++c) {
            int cid = (sx[a]*g.ny + sy[b])*g.nz + sz[c];
            int j0 = cell_start[cid], j1 = cell_start[cid+1];
            if (AUTO) { if (j1 <= i+1) continue; if (j0 < i+1) j0 = i+1; }
            for (int j=j0;j<j1;++j) {
                double dx,dy,dz;
                disp<PERIODIC>(x1,y1,z1,bx[j],by[j],bz[j],g,dx,dy,dz);
                double rsq = dx*dx + dy*dy + dz*dz;
                if (rsq >= minsepsq && rsq < maxsepsq) {
                    double r = sqrt(rsq);
                    double logr = log(r);
                    int k = int((logr - logminsep) / binsize);
                    if (k >= nbins) k = nbins-1;
                    if (k < 0) k = 0;
                    double ww = w1 * bw[j];
                    atomicAdd(&sh[k], 1.);
                    atomicAdd(&sh[nbins+k], ww);
                    atomicAdd(&sh[2*nbins+k], ww*r);
                    atomicAdd(&sh[3*nbins+k], ww*logr);
                }
            }
        }
    }
    __syncthreads();
    for (int t=threadIdx.x; t<4*nbins; t+=blockDim.x) if (sh[t] != 0.) atomicAdd(&out[t], sh[t]);
}

// ---------------------------------------------------------------------------------------
// 3-point: neighbour lists.  For each center c (A, indices [c0, c0+nc)) find all B points
// with minsep <= d < maxsep (d = sqrt(|wrap(p_b - p_c)|^2), TreeCorr's isTriangleInRange
// test) and log bin k = int(floor((log d - logminsep)/binsize)).
// pass 0: count per (center, k).   pass 1: fill sorted by (center,k).
// EXCL: A and B are the same catalog: skip j == c.
// ---------------------------------------------------------------------------------------
struct ListOut {
    int* idx;            // index into B (sorted arrays)
    float* vx; float* vy; float* vz;  // displacement (mixed mode)
    float* roff; float* lroff;        // d - ref_k, log d - logref_k
    double* dvx; double* dvy; double* dvz;  // displacement (double mode)
    double* dd; double* dlogd;              // d, log d (double mode)
};

template <bool PERIODIC, bool EXCL, int PASS, bool DBL>
__global__ void nbr_kernel(const double* __restrict__ ax, const double* __restrict__ ay,
                           const double* __restrict__ az, int c0, int nc,
                           const double* __restrict__ bx, const double* __restrict__ by,
                           const double* __restrict__ bz,
                           const int* __restrict__ cell_start, Geom g,
                           double minsep, double maxsep, double logminsep, double binsize,
                           int nbins, const double* __restrict__ refd,
                           const double* __restrict__ reflogd,
                           int* counts, const long long* __restrict__ offsets, int* cursor,
                           ListOut L)
{
    int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= nc) return;
    int i = c0 + ci;
    const double x1 = ax[i], y1 = ay[i], z1 = az[i];
    int sx[MAXST], sy[MAXST], sz[MAXST];
    int nsx = stencil1<PERIODIC>(cellcoord<PERIODIC>(x1, g.x0, g.cx, g.nx, g.Lx), g.nx, g.m, sx);
    int nsy = stencil1<PERIODIC>(cellcoord<PERIODIC>(y1, g.y0, g.cy, g.ny, g.Ly), g.ny, g.m, sy);
    int nsz = stencil1<PERIODIC>(cellcoord<PERIODIC>(z1, g.z0, g.cz, g.nz, g.Lz), g.nz, g.m, sz);
    for (int a=0;a<nsx;++a) for (int b=0;b<nsy;++b) for (int c=0;c<nsz;++c) {
        int cid = (sx[a]*g.ny + sy[b])*g.nz + sz[c];
        int j0 = cell_start[cid], j1 = cell_start[cid+1];
        for (int j=j0;j<j1;++j) {
            if (EXCL && j == i) continue;
            double dx,dy,dz;
            disp<PERIODIC>(x1,y1,z1,bx[j],by[j],bz[j],g,dx,dy,dz);
            double dsq = dx*dx + dy*dy + dz*dz;
            double d = sqrt(dsq);
            if (d < minsep || d >= maxsep) continue;
            double logd = log(d);
            int k = int(floor((logd - logminsep) / binsize));
            if (k == nbins) --k;
            if (k < 0 || k >= nbins) continue;   // cannot happen given the range test
            long long slot = (long long)ci * nbins + k;
            if (PASS == 0) {
                atomicAdd(&counts[slot], 1);
            } else {
                long long p = offsets[slot] + atomicAdd(&cursor[slot], 1);
                L.idx[p] = j;
                if (!DBL) {   // double mode recomputes everything from the positions
                    ((float4*)L.vx)[p] = make_float4((float)dx, (float)dy, (float)dz,
                                                     (float)(d - refd[k]));
                    L.lroff[p] = (float)(logd - reflogd[k]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// 3-point triplet kernel.
//
// grid.x : bin pair index (kb, kl) from the pair table;  grid.y : chunk of centers.
// For every center c in the chunk, loops over the neighbour pairs (j in list B bin kb,
// l in list C bin kl).  SAME: lists B and C are the same list -> unordered pairs
// (kb <= kl, and j < l when kb == kl).
//
// Orientation (TreeCorr process111/O=1, swap_23): initial (c2,c3) = (j,l);
//   if CCW(p1,p3,p2) [ cross(v_l, v_j).p_c < 0 ]  ->  P2=j, P3=l   -> (k2,k3) = (kl,kb), o=0
//   else swap                                       ->  P2=l, P3=j   -> (k2,k3) = (kb,kl), o=1
//
// Local histogram: 2 orientations x nphi bins.  Sums are accumulated as 64-bit fixed point
// (exact, order independent) of (value - reference) and flushed to global fp64.
// ---------------------------------------------------------------------------------------

// Between two flushes a bin receives at most TILE terms, each |q| <= 2^42 in fixed point,
// so the int64 sums stay below 2^62.
#define TILE (1LL<<20)
// mixed mode: fp32 shared sums, flushed to fp64 at least every FTILE pairs
#define FTILE (1LL<<18)
#define NQ 7   // fixed-point sums: d1, logd1, d2, logd2, d3, logd3, phi

struct TriParams {
    double minsep, maxsep, logminsep, binsize;
    double minphi, maxphi, phibinsize;
    int nbins, nphi;
    double guard;            // angular guard band (radians) for the fp32 fast path
    double qscale[NQ];       // fixed-point scales
    double qinv[NQ];
    double qbound[NQ];       // |term| bound for the fixed-point path; above -> direct global add
};

struct TriOut {
    u64* ntri;
    double* weight;
    double* s[NQ];           // meand1, meanlogd1, meand2, meanlogd2, meand3, meanlogd3, meanphi
    u64* nexact;             // diagnostics: number of triplets that went through the exact path
};

struct ListIn {
    const int* idx;
    const float* vx; const float* vy; const float* vz;
    const float* roff; const float* lroff;
    const double* dvx; const double* dvy; const double* dvz;
    const double* dd; const double* dlogd;
    const long long* offsets;    // (nc*nbins + 1) flat offsets
    const double* px; const double* py; const double* pz; const double* pw;   // catalog
};

__device__ __forceinline__ void fx_add(i64* slot, double v, double scale)
{
    i64 q = __double2ll_rn(v * scale);
    atomicAdd((u64*)slot, (u64)q);
}

template <bool UNITW, bool DBL>
__device__ void flush_hist(i64* hq, double* hw, unsigned int* hc, int NB, int nphi, int nb,
                           int kb, int kl, const double* refd, const double* reflogd,
                           const double* refd1, const double* reflogd1,
                           const TriParams& P, TriOut& out)
{
    __syncthreads();
    for (int t=threadIdx.x; t<NB; t+=blockDim.x) {
        const unsigned int cnt = hc[t];
        if (cnt == 0u) continue;
        const int o = t / nphi, kphi = t - o*nphi;
        const int k2 = o ? kb : kl;
        const int k3 = o ? kl : kb;
        const int gidx = (k2*nb + k3)*nphi + kphi;
        const double wsum = UNITW ? (double)cnt : hw[t];
        atomicAdd(&out.ntri[gidx], (u64)cnt);
        atomicAdd(&out.weight[gidx], wsum);
        double ref[NQ];
        ref[0] = refd1[gidx]; ref[1] = reflogd1[gidx];
        ref[2] = refd[k2]; ref[3] = reflogd[k2];
        ref[4] = refd[k3]; ref[5] = reflogd[k3];
        ref[6] = P.minphi + (kphi + 0.5) * P.phibinsize;
        #pragma unroll
        for (int iq=0; iq<NQ; ++iq) {
            const double qsum = DBL ? (double)hq[iq*NB + t] * P.qinv[iq]
                                    : (double)((float*)hq)[iq*NB + t];
            const double s = qsum + wsum * ref[iq];
            atomicAdd(&out.s[iq][gidx], s);
            if (DBL) hq[iq*NB + t] = 0; else ((float*)hq)[iq*NB + t] = 0.f;
        }
        hc[t] = 0u; hw[t] = 0.;
    }
    __syncthreads();
}

// Exact fp64 evaluation of one triangle with TreeCorr's leaf formulas (Corr3.cpp process111 /
// BinType.h LogSAS::stop111, singleBin, isTriangleInRange).  Returns false if not accumulated.
template <bool PERIODIC>
__device__ __noinline__ bool exact_triangle(
    double pcx, double pcy, double pcz, double xj, double yj, double zj,
    double xl, double yl, double zl, const Geom& g, const TriParams& P,
    bool& swap, int& kphi, double& phi, double& d1, double& logd1,
    double& d2, double& logd2, double& d3, double& logd3)
{
    double vjx,vjy,vjz,vlx,vly,vlz;
    disp<PERIODIC>(pcx,pcy,pcz,xj,yj,zj,g,vjx,vjy,vjz);   // Displacement(p1, pj)
    disp<PERIODIC>(pcx,pcy,pcz,xl,yl,zl,g,vlx,vly,vlz);   // Displacement(p1, pl)
    // initial (c2,c3) = (j,l).  CCW(p1, p3, p2) = cross(r13, r12).dot(p1) < 0  -> no swap
    const double cx = vly*vjz - vlz*vjy;
    const double cy = vlz*vjx - vlx*vjz;
    const double cz = vlx*vjy - vly*vjx;
    swap = !((cx*pcx + cy*pcy + cz*pcz) < 0.);
    double x2,y2,z2,x3,y3,z3;
    if (!swap) { x2=xj;y2=yj;z2=zj; x3=xl;y3=yl;z3=zl; }
    else       { x2=xl;y2=yl;z2=zl; x3=xj;y3=yj;z3=zj; }
    double ux,uy,uz;
    disp<PERIODIC>(x2,y2,z2,x3,y3,z3,g,ux,uy,uz);         // Displacement(p2, p3)
    const double d1sq = ux*ux + uy*uy + uz*uz;
    double d2sq, d3sq;
    if (!swap) { d2sq = vlx*vlx + vly*vly + vlz*vlz; d3sq = vjx*vjx + vjy*vjy + vjz*vjz; }
    else       { d2sq = vjx*vjx + vjy*vjy + vjz*vjz; d3sq = vlx*vlx + vly*vly + vlz*vlz; }
    if (d1sq == 0.) return false;                         // stop111: coincident leaves
    d3 = sqrt(d3sq); d2 = sqrt(d2sq);
    const double cosphi = (d2sq + d3sq - d1sq) / (2*d2*d3);
    if (cosphi > -1 && cosphi < 1) phi = acos(cosphi);
    else if (cosphi <= -1) phi = M_PI;
    else phi = 0.;
    if (phi < P.minphi || phi >= P.maxphi) return false;
    kphi = int(floor((phi - P.minphi) / P.phibinsize));
    if (kphi >= P.nphi) --kphi;
    d1 = sqrt(d1sq);
    logd1 = log(d1);
    logd2 = log(d2); logd3 = log(d3);
    return true;
}

// Tile size for staging neighbour lists in shared memory.
#define LT 256

template <bool PERIODIC, bool SAME, bool DBL, bool UNITW>
__global__ void __launch_bounds__(256, 3)
tri_kernel(const int2* __restrict__ binpairs,
           const double* __restrict__ ax, const double* __restrict__ ay,
           const double* __restrict__ az, const double* __restrict__ aw,
           int c0, int nc, int chunk,
           ListIn B, ListIn C, Geom g, TriParams P,
           const double* __restrict__ refd, const double* __restrict__ reflogd,
           const double* __restrict__ refd1,    // [nbins*nbins*nphi] d1 reference
           const double* __restrict__ reflogd1,
           TriOut out)
{
    const int2 bp = binpairs[blockIdx.x];
    const int kb = bp.x, kl = bp.y;
    const int nphi = P.nphi, nb = P.nbins;
    const int NB = 2*nphi;

    extern __shared__ unsigned char smem[];
    i64* hq = (i64*)smem;                                // NQ*NB
    double* hw = (double*)(hq + NQ*NB);                  // NB
    unsigned int* hc = (unsigned int*)(hw + NB);         // NB
    float* fref = (float*)(hc + NB);                     // 2*nphi: d1 ref, log d1 ref
    // staged neighbour tiles (mixed mode): float4 (vx,vy,vz,roff) + lroff
    float4* sj = (float4*)(((uintptr_t)(fref + 2*nphi) + 15) & ~(uintptr_t)15);
    float4* sl = sj + LT;
    float* sjl = (float*)(sl + LT);
    float* sll = sjl + LT;

    for (int t=threadIdx.x; t<NQ*NB; t+=blockDim.x) hq[t] = 0;
    for (int t=threadIdx.x; t<NB; t+=blockDim.x) { hw[t] = 0.; hc[t] = 0u; }
    for (int t=threadIdx.x; t<nphi; t+=blockDim.x) {
        fref[t] = (float)refd1[(kb*nb + kl)*nphi + t];
        fref[nphi + t] = (float)reflogd1[(kb*nb + kl)*nphi + t];
    }
    __syncthreads();

    // fp32 constants for the fast path
    const float fminphi = (float)P.minphi, fmaxphi = (float)P.maxphi;
    const float fbin = (float)P.phibinsize, finv = (float)(1. / P.phibinsize);
    const float fgx = (float)(P.guard / P.phibinsize), fguard = (float)P.guard;
    float fscale[NQ], fbound[NQ];
    #pragma unroll
    for (int iq=0; iq<NQ; ++iq) { fscale[iq] = (float)P.qscale[iq]; fbound[iq] = (float)P.qbound[iq]; }

    const int cbeg = blockIdx.y * chunk;
    const int cend = min(nc, cbeg + chunk);
    long long since_flush = 0;
    u64 my_exact = 0;
    const float4* B4 = (const float4*)B.vx;
    const float4* C4 = (const float4*)C.vx;

    for (int ci = cbeg; ci < cend; ++ci) {
        const long long sB = B.offsets[(long long)ci*nb + kb];
        const long long eB = B.offsets[(long long)ci*nb + kb + 1];
        const long long sC = C.offsets[(long long)ci*nb + kl];
        const long long eC = C.offsets[(long long)ci*nb + kl + 1];
        const int nj = (int)(eB - sB), nl = (int)(eC - sC);
        if (nj == 0 || nl == 0) continue;
        const int ic = c0 + ci;
        const double pcx = ax[ic], pcy = ay[ic], pcz = az[ic];
        const double wc = UNITW ? 1. : aw[ic];
        const float fpx = (float)pcx, fpy = (float)pcy, fpz = (float)pcz;
        const float pnorm = sqrtf(fpx*fpx + fpy*fpy + fpz*fpz);

        for (int jt0 = 0; jt0 < nj; jt0 += LT) {
        const int njt = min(LT, nj - jt0);
        for (int lt0 = 0; lt0 < nl; lt0 += LT) {
            const int nlt = min(LT, nl - lt0);
            if (SAME && kb == kl && lt0 + nlt <= jt0 + 1) continue;   // no b > a in this tile
            const int ntile = njt * nlt;
            // uniform flush decision (since_flush is identical in all threads)
            if (since_flush + ntile > (DBL ? TILE : FTILE)) {
                flush_hist<UNITW, DBL>(hq, hw, hc, NB, nphi, nb, kb, kl, refd, reflogd, refd1, reflogd1, P, out);
                since_flush = 0;
            }
            if (!DBL) {
                __syncthreads();
                for (int t = threadIdx.x; t < njt; t += blockDim.x) {
                    sj[t] = B4[sB + jt0 + t]; sjl[t] = B.lroff[sB + jt0 + t];
                }
                for (int t = threadIdx.x; t < nlt; t += blockDim.x) {
                    sl[t] = C4[sC + lt0 + t]; sll[t] = C.lroff[sC + lt0 + t];
                }
                __syncthreads();
            }
            const float inv_nlt = 1.f / (float)nlt;
            for (int u = threadIdx.x; u < ntile; u += blockDim.x) {
                int a = (int)((float)u * inv_nlt);
                int b = u - a * nlt;
                if (b < 0) { --a; b += nlt; } else if (b >= nlt) { ++a; b -= nlt; }
                if (SAME && kb == kl && lt0 + b <= jt0 + a) continue;
                const long long pj = sB + jt0 + a, pl = sC + lt0 + b;

                bool swap = false; int kphi = 0;
                bool exact = DBL;
                float qf[NQ];
                double qd[NQ];

                if (!DBL) {
                    const float4 J = sj[a], Lv = sl[b];
                    const float cxx = Lv.y*J.z - Lv.z*J.y;          // cross(v_l, v_j)
                    const float cxy = Lv.z*J.x - Lv.x*J.z;
                    const float cxz = Lv.x*J.y - Lv.y*J.x;
                    const float crn = sqrtf(cxx*cxx + cxy*cxy + cxz*cxz);
                    const float dot = J.x*Lv.x + J.y*Lv.y + J.z*Lv.z;
                    const float ori = cxx*fpx + cxy*fpy + cxz*fpz;
                    const float rj2 = J.x*J.x + J.y*J.y + J.z*J.z;
                    const float rl2 = Lv.x*Lv.x + Lv.y*Lv.y + Lv.z*Lv.z;
                    const float rr = sqrtf(rj2 * rl2);
                    const float fphi = atan2f(crn, dot);
                    const float ex = Lv.x - J.x, ey = Lv.y - J.y, ez = Lv.z - J.z;
                    const float d1sq = ex*ex + ey*ey + ez*ez;
                    const float x = (fphi - fminphi) * finv;
                    const float fk = floorf(x);
                    const float dedge = fminf(x - fk, fk + 1.f - x);
                    if (dedge < fgx ||
                        fabsf(fphi - fminphi) < fguard || fabsf(fphi - fmaxphi) < fguard ||
                        fabsf(ori) < 1.e-5f * rr * pnorm + 1.e-30f ||
                        crn < 1.e-4f * rr ||
                        d1sq < 1.e-6f * (rj2 + rl2)) {
                        exact = true;
                    } else {
                        if (fphi < fminphi || fphi >= fmaxphi) continue;
                        kphi = (int)fk;
                        if (kphi >= nphi) kphi = nphi-1;
                        swap = !(ori < 0.f);
                        const float fd1 = sqrtf(d1sq);
                        const float rjo = J.w, ljo = sjl[a];
                        const float rlo = Lv.w, llo = sll[b];
                        qf[0] = fd1 - fref[kphi];
                        qf[1] = logf(fd1) - fref[nphi + kphi];
                        // !swap: P2 = j -> d3 = r_j, d2 = r_l
                        qf[2] = swap ? rjo : rlo;
                        qf[3] = swap ? ljo : llo;
                        qf[4] = swap ? rlo : rjo;
                        qf[5] = swap ? llo : ljo;
                        qf[6] = fphi - (fminphi + ((float)kphi + 0.5f) * fbin);
                    }
                }

                int k2, k3, gidx;
                if (exact) {
                    ++my_exact;
                    const int jb = B.idx[pj], lc = C.idx[pl];
                    double phi, d1, logd1, d2, logd2, d3, logd3;
                    if (!exact_triangle<PERIODIC>(pcx, pcy, pcz, B.px[jb], B.py[jb], B.pz[jb],
                                                  C.px[lc], C.py[lc], C.pz[lc], g, P, swap, kphi,
                                                  phi, d1, logd1, d2, logd2, d3, logd3)) continue;
                    k2 = swap ? kb : kl; k3 = swap ? kl : kb;
                    gidx = (k2*nb + k3)*nphi + kphi;
                    qd[0] = d1 - refd1[gidx];
                    qd[1] = logd1 - reflogd1[gidx];
                    qd[2] = d2 - refd[k2];
                    qd[3] = logd2 - reflogd[k2];
                    qd[4] = d3 - refd[k3];
                    qd[5] = logd3 - reflogd[k3];
                    qd[6] = phi - (P.minphi + (kphi + 0.5) * P.phibinsize);
                } else {
                    k2 = swap ? kb : kl; k3 = swap ? kl : kb;
                    gidx = (k2*nb + k3)*nphi + kphi;
                }
                const int hb = (swap ? nphi : 0) + kphi;

                double www = 1.;
                if (!UNITW) www = wc * B.pw[B.idx[pj]] * C.pw[C.idx[pl]];

                atomicAdd(&hc[hb], 1u);
                if (!UNITW) atomicAdd(&hw[hb], www);
                if (DBL) {
                    // exact, order-independent 64-bit fixed point
                    #pragma unroll
                    for (int iq=0; iq<NQ; ++iq) {
                        const double v = www * qd[iq];
                        if (fabs(v) <= P.qbound[iq])
                            atomicAdd((u64*)&hq[iq*NB + hb], (u64)__double2ll_rn(v * P.qscale[iq]));
                        else atomicAdd(&out.s[iq][gidx], v);     // rare: outside fixed-point range
                    }
                } else {
                    // fp32 shared sums of small offsets (value - bin reference), flushed to fp64
                    float* hf = (float*)hq;
                    if (exact) {
                        #pragma unroll
                        for (int iq=0; iq<NQ; ++iq) qf[iq] = (float)qd[iq];
                    }
                    const float fw = (float)www;
                    #pragma unroll
                    for (int iq=0; iq<NQ; ++iq) {
                        const float v = UNITW ? qf[iq] : fw * qf[iq];
                        if (fabsf(v) <= fbound[iq]) atomicAdd(&hf[iq*NB + hb], v);
                        else atomicAdd(&out.s[iq][gidx], (double)v);
                    }
                }
            }   // pair loop
            since_flush += ntile;
        }   // l tiles
        }   // j tiles
    }   // centre loop
    flush_hist<UNITW, DBL>(hq, hw, hc, NB, nphi, nb, kb, kl, refd, reflogd, refd1, reflogd1, P, out);
    if (my_exact) atomicAdd(out.nexact, my_exact);
}

// =======================================================================================
// Legendre / spherical-harmonic (Slepian & Eisenstein 2015) mode.
//
// For every (centre, radial bin) slot:  a_lm = sum_j w_j Y_lm(r_hat_j)   (m >= 0 only)
// plus the moments  sum w, sum w d, sum w log d, sum w^2, sum w^2 d, sum w^2 log d.
// Y_lm uses fully normalised associated Legendre functions (stable 3-term recursion).
// Thread layout: lane m = t % (lmax+1) handles one m for all l >= m; G = blockDim/(lmax+1)
// neighbour groups stride over the slot's neighbours; results reduced in shared memory.
// =======================================================================================
template <bool PERIODIC, bool DBL, bool UNITW, int LM>
__global__ void __launch_bounds__(128)
alm_kernel(ListIn L, const double* __restrict__ ax, const double* __restrict__ ay,
           const double* __restrict__ az, int c0, int nslots, int nb, int lmax,
           const double* __restrict__ refd, const double* __restrict__ ca,
           const double* __restrict__ cb, Geom g, double* alm_out, double* mom_out)
{
    typedef typename SelT<DBL>::type T;
    const int slot = blockIdx.x;
    if (slot >= nslots) return;
    const long long s = L.offsets[slot], e = L.offsets[slot + 1];
    if (s == e) return;                       // output pre-zeroed
    const int L1 = lmax + 1;
    const int nlm = L1 * (L1 + 1) / 2;
    const int G = blockDim.x / L1;
    const int t = threadIdx.x;
    const int m = t % L1, grp = t / L1;
    const bool active = grp < G;
    const int ci = slot / nb, k = slot - ci * nb;

    extern __shared__ unsigned char smem_raw[];
    T* part = (T*)smem_raw;                   // G * 2*nlm partial sums (one row per group)
    T* red = part + (long long)G * 2 * nlm;   // 6 moments
    for (int i = t; i < G * 2 * nlm + 6; i += blockDim.x) part[i] = (T)0;
    __syncthreads();
    T* acc = part + (long long)grp * 2 * nlm; // rows owned by this group; column m by this thread
    T mw = 0, mwd = 0, mwl = 0, mw2 = 0, mw2d = 0, mw2l = 0;

    double pcx = 0., pcy = 0., pcz = 0.;
    if (DBL) { const int ic = c0 + ci; pcx = ax[ic]; pcy = ay[ic]; pcz = az[ic]; }
    const T inv4pi = (T)(0.25 / M_PI);

    if (active) {
        for (long long n = s + grp; n < e; n += G) {
            T x, y, z, d;
            if (DBL) {
                const int j = L.idx[n];
                double dx, dy, dz;
                disp<PERIODIC>(pcx, pcy, pcz, L.px[j], L.py[j], L.pz[j], g, dx, dy, dz);
                x = dx; y = dy; z = dz;
                d = sqrt(dx*dx + dy*dy + dz*dz);
            } else {
                const float4 v = ((const float4*)L.vx)[n];
                x = v.x; y = v.y; z = v.z;
                d = sqrtf(x*x + y*y + z*z);
            }
            const T w = UNITW ? (T)1 : (T)L.pw[L.idx[n]];
            const T rho = sqrt(x*x + y*y);
            const T ct = z / d, st = rho / d;
            T cph = 1, sph = 0;
            if (rho > 0) { cph = x / rho; sph = y / rho; }
            if (m == 0) {
                const T ld = log(d);
                mw += w; mwd += w*d; mwl += w*ld;
                mw2 += w*w; mw2d += w*w*d; mw2l += w*w*ld;
            }
            // e^{i m phi}
            T er = 1, ei = 0;
            for (int i = 0; i < m; ++i) { const T tr = er*cph - ei*sph; ei = er*sph + ei*cph; er = tr; }
            // P_mm (normalised, Condon-Shortley phase)
            T pmm = sqrt(inv4pi);
            for (int i = 1; i <= m; ++i) pmm *= -sqrt((T)(2*i + 1) / (T)(2*i)) * st;
            const T wr = w * er, wi = w * ei;
            { const int lm = m*(m+1)/2 + m; acc[2*lm] += pmm * wr; acc[2*lm+1] += pmm * wi; }
            if (m < lmax) {
                T p0 = pmm;
                T p1 = sqrt((T)(2*m + 3)) * ct * pmm;
                { const int lm = (m+1)*(m+2)/2 + m; acc[2*lm] += p1 * wr; acc[2*lm+1] += p1 * wi; }
                for (int l = m + 2; l <= lmax; ++l) {
                    const int lm = l*(l+1)/2 + m;
                    const T p2 = (T)ca[lm] * (ct * p1 - (T)cb[lm] * p0);
                    acc[2*lm] += p2 * wr; acc[2*lm+1] += p2 * wi;
                    p0 = p1; p1 = p2;
                }
            }
        }
        if (m == 0) {
            atomicAdd(&red[0], mw);  atomicAdd(&red[1], mwd);
            atomicAdd(&red[2], mwl); atomicAdd(&red[3], mw2);
            atomicAdd(&red[4], mw2d); atomicAdd(&red[5], mw2l);
        }
    }
    __syncthreads();
    for (int i = t; i < 2 * nlm; i += blockDim.x) {
        double v = 0.;
        for (int gg = 0; gg < G; ++gg) v += (double)part[(long long)gg * 2 * nlm + i];
        alm_out[(long long)slot * 2 * nlm + i] = v;
    }
    for (int i = t; i < 6; i += blockDim.x) mom_out[(long long)slot * 6 + i] = (double)red[i];
}
