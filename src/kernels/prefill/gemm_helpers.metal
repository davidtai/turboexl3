struct Arows { const device half* r[4]; };
template <uint N, uint WMASK, int CB, int JOBS, int N16, int LDB, int P, int FB, int IR, int DF>
inline void exl3pf_stage(const device uint* wb, int s, thread const ulong* woff, thread const int* boff, threadgroup half* Bs, uint qb, uint ga, uint lp, uint lnx) {
    for (int jj = 0; jj < JOBS; ++jj) {
        if (P & 1) {
            const half k0 = half(float(s & 7) * 0.125f);
            if (FB != 0) { *reinterpret_cast<threadgroup vec<half, 8>*>(Bs + boff[jj] + lnx * 8u) = vec<half, 8>(k0); continue; }
            threadgroup half* d = Bs + boff[jj];
            *reinterpret_cast<threadgroup half2*>(d) = half2(k0, k0);
            *reinterpret_cast<threadgroup half2*>(d + 8) = half2(k0, k0);
            *reinterpret_cast<threadgroup half2*>(d + 8 * LDB) = half2(k0, k0);
            *reinterpret_cast<threadgroup half2*>(d + 8 * LDB + 8) = half2(k0, k0);
            continue;
        }
        const device uint* tw = wb + ulong(s) * ulong(4 * N16) * ulong(N / 2u) + woff[jj];
        const auto ww = exl3_tile_load<N, IR>(tw, qb, ga, lp);
        const half v0 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 0, IR>(ww, lp));
        const half v1 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 1, IR>(ww, lp));
        const half v2 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 2, IR>(ww, lp));
        const half v3 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 3, IR>(ww, lp));
        const half v4 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 4, IR>(ww, lp));
        const half v5 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 5, IR>(ww, lp));
        const half v6 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 6, IR>(ww, lp));
        const half v7 = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 7, IR>(ww, lp));
        if ((P & 32) != 0 && FB != 0) {
            // C14 cost probe (outputs wrong): a 128-point Hadamard's butterflies over the decoded weights, 3 levels in the
            // lane's 8 values and 4 across lanes (the floor of folding the input rotation into the weights)
            half v[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
            for (int h = 1; h < 8; h <<= 1) for (int i = 0; i < 8; ++i) if ((i & h) == 0) { const half a = v[i], b = v[i + h]; v[i] = a + b; v[i + h] = a - b; }
            for (ushort m = 1; m < 16; m <<= 1) for (int i = 0; i < 8; ++i) { const half o = simd_shuffle_xor(v[i], m); v[i] = (lnx & m) != 0u ? o - v[i] : v[i] + o; }
            *reinterpret_cast<threadgroup vec<half, 8>*>(Bs + boff[jj] + lnx * 8u) = vec<half, 8>(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
            continue;
        }
        if (FB != 0) { *reinterpret_cast<threadgroup vec<half, 8>*>(Bs + boff[jj] + lnx * 8u) = vec<half, 8>(v0, v1, v2, v3, v4, v5, v6, v7); continue; }
        threadgroup half* d = Bs + boff[jj];
        *reinterpret_cast<threadgroup half2*>(d) = half2(v0, v1);
        *reinterpret_cast<threadgroup half2*>(d + 8) = half2(v2, v3);
        *reinterpret_cast<threadgroup half2*>(d + 8 * LDB) = half2(v4, v5);
        *reinterpret_cast<threadgroup half2*>(d + 8 * LDB + 8) = half2(v6, v7);
    }
}
// FB: the stage holds each NAX lane's B^T fragment (8 halfs, 16 bytes) contiguously, written by the lane whose plan decodes
// it; the A fragments are read in the same slot order (k 2q, 2q + 1, 2q + 8, 2q + 9 of each row, the register path's rule):
// two half2 per fragment row and 16-deep slice from natural-order rows, or from rows stored in that slot order (AL 1: one
// 16-byte load per fragment row and 32-deep step; AL 2: one 8-byte load per fragment row and 16-deep slice). H: a simdgroup
// with at most 16 rows (sgp <= 16) loads and multiplies only its first fragment row (its rows 16..31 are past the tile and
// never stored)
template <int LDB, int KDC, int P, int U, int FB, int AL, int H>
inline void exl3pf_mma(thread NAXTile<float, 2, 2>& Dt, const device half* xa, thread const Arows& arw, int s, const threadgroup half* Bs, int tn, bool full, int sgp, uint L) {
    if (FB != 0) {
        const bool top = H != 0 && sgp <= 16;
        STEEL_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            NAXTile<half, 2, 2> At;
            NAXTile<half, 2, 2> Bt;
            if (AL == 1) {
                const int k32 = s * 64 + i * 32;
                if (P & 4) At.clear();
                else {
                    exl3pf_afrag2(At.frag_at(0, 0), At.frag_at(0, 1), arw.r[0] + k32, arw.r[1] + k32);
                    if (!top) exl3pf_afrag2(At.frag_at(1, 0), At.frag_at(1, 1), arw.r[2] + k32, arw.r[3] + k32);
                }
            }
            STEEL_PRAGMA_UNROLL
            for (int kk = 0; kk < 2; ++kk) {
                const int k0 = s * 64 + i * 32 + kk * 16;
                if (AL != 1) {
                    if (P & 4) { At.frag_at(0, kk) = vec<half, 8>(0); At.frag_at(1, kk) = vec<half, 8>(0); }
                    else if (AL == 2) { exl3pf_afrag4(At.frag_at(0, kk), arw.r[0] + k0, arw.r[1] + k0); if (!top) exl3pf_afrag4(At.frag_at(1, kk), arw.r[2] + k0, arw.r[3] + k0); }
                    else { exl3pf_afrag(At.frag_at(0, kk), arw.r[0] + k0, arw.r[1] + k0); if (!top) exl3pf_afrag(At.frag_at(1, kk), arw.r[2] + k0, arw.r[3] + k0); }
                }
                STEEL_PRAGMA_UNROLL
                for (int nn = 0; nn < 2; ++nn) {
                    if (P & 8) Bt.frag_at(nn, kk) = vec<half, 8>(0);
                    else Bt.frag_at(nn, kk) = *reinterpret_cast<const threadgroup vec<half, 8>*>(Bs + ((tn / 16 + nn) * 4 + 2 * i + kk) * 256 + L * 8u);
                }
            }
            if (top) exl3pf_mma_top(Dt, At, Bt);
            else tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
        }
        return;
    }
    if (U != 0) {
        NAXTile<half, 2, 2> At0, At1, Bt0, Bt1;
        if (P & 4) { At0.clear(); At1.clear(); }
        else if (full) { At0.load(xa + s * 64, KDC); At1.load(xa + s * 64 + 32, KDC); }
        else { At0.load_safe(xa + s * 64, KDC, short2(32, sgp)); At1.load_safe(xa + s * 64 + 32, KDC, short2(32, sgp)); }
        if (P & 8) { Bt0.clear(); Bt1.clear(); }
        else { Bt0.template load<half, LDB, 1>(Bs + tn * LDB); Bt1.template load<half, LDB, 1>(Bs + tn * LDB + 32); }
        tile_matmad_nax(Dt, At0, metal::bool_constant<false>{}, Bt0, metal::bool_constant<true>{});
        tile_matmad_nax(Dt, At1, metal::bool_constant<false>{}, Bt1, metal::bool_constant<true>{});
        return;
    }
    STEEL_PRAGMA_NO_UNROLL
    for (int i = 0; i < 2; ++i) {
        NAXTile<half, 2, 2> At;
        NAXTile<half, 2, 2> Bt;
        volatile int compiler_barrier;
        if (P & 4) At.clear();
        else if (full) At.load(xa + s * 64 + i * 32, KDC);
        else At.load_safe(xa + s * 64 + i * 32, KDC, short2(32, sgp));
        if (P & 8) Bt.clear();
        else Bt.template load<half, LDB, 1>(Bs + tn * LDB + i * 32);
        tile_matmad_nax(Dt, At, metal::bool_constant<false>{}, Bt, metal::bool_constant<true>{});
        (void)compiler_barrier;
    }
}
