// the deferred MUL1 map on a 32 x 32 accumulator block: lane L holds rows fm + 8 h + 16 mm (BaseNAXFrag::get_coord); each
// row's input sum is its NB per-128-block partials (prep / mid); rows past `rows` are never stored
inline void exl3pf_affine(thread NAXTile<float, 2, 2>& Dt, const device float* xsr, int nb, int rows, uint L) {
    const float MA = float(as_type<half>(ushort(0x1EEEu)));
    const float MC = 1024.0f * MA + float(as_type<half>(ushort(0xC931u)));
    const int fm = int(((L >> 2u) & 4u) | ((L >> 1u) & 3u));
    for (short mm = 0; mm < 2; ++mm) {
        for (short hh = 0; hh < 2; ++hh) {
            const int r = fm + 8 * hh + 16 * mm;
            float sx = 0.0f;
            if (r < rows) for (int q = 0; q < nb; ++q) sx += xsr[ulong(r) * ulong(nb) + q];
            for (short nn = 0; nn < 2; ++nn) {
                for (short jj = 0; jj < 4; ++jj) Dt.frag_at(mm, nn)[hh * 4 + jj] = fma(Dt.frag_at(mm, nn)[hh * 4 + jj], MA, MC * sx);
            }
        }
    }
}
// a decoded weight for the GEMMs: the codebook value, or (DF, MUL1) the codeword's byte sum t in [0, 1020], exact in f16; the
// epilogue then applies w = A t + C as A acc + C sum(x) (A = half 0x1EEE, C = 1024 A + half 0xC931)
template <int CB, int DF> inline half exl3pf_w(uint cw) {
    if (DF != 0) return exl3_mul1_th(cw);
    return exl3_decode<CB>(cw);
}
inline void exl3pf_afrag(thread vec<half, 8>& f, const device half* r0, const device half* r1) {
    const half2 a0 = *reinterpret_cast<const device half2*>(r0);
    const half2 a1 = *reinterpret_cast<const device half2*>(r0 + 8);
    const half2 a2 = *reinterpret_cast<const device half2*>(r1);
    const half2 a3 = *reinterpret_cast<const device half2*>(r1 + 8);
    f[0] = a0.x; f[1] = a0.y; f[2] = a1.x; f[3] = a1.y; f[4] = a2.x; f[5] = a2.y; f[6] = a3.x; f[7] = a3.y;
}
// ALAY 1: rows stored with each 32-block in the fragments' slot order (`turboexl3_alay`), r0 / r1 at the block's half 8q of the
// fragment's two rows: halfs 0..3 are slots 4q..4q + 3 of the first 16-deep slice (f0), 4..7 those of the second (f1) - the
// values exl3pf_afrag reads at k0 and k0 + 16 from natural-order rows, one 16-byte load per row instead of four half2
inline void exl3pf_afrag2(thread vec<half, 8>& f0, thread vec<half, 8>& f1, const device half* r0, const device half* r1) {
    const vec<half, 8> a = *reinterpret_cast<const device vec<half, 8>*>(r0);
    const vec<half, 8> b = *reinterpret_cast<const device vec<half, 8>*>(r1);
    f0[0] = a[0]; f0[1] = a[1]; f0[2] = a[2]; f0[3] = a[3]; f0[4] = b[0]; f0[5] = b[1]; f0[6] = b[2]; f0[7] = b[3];
    f1[0] = a[4]; f1[1] = a[5]; f1[2] = a[6]; f1[3] = a[7]; f1[4] = b[4]; f1[5] = b[5]; f1[6] = b[6]; f1[7] = b[7];
}
// ALAY 2: rows stored with each 16-block in the fragments' slot order (`turboexl3_alay16`, dense.zig's LANE_ORDER rule), r0 / r1
// at the slice's half 4q of the fragment's two rows: one 8-byte load per row, each straight into one half of the fragment
inline void exl3pf_afrag4(thread vec<half, 8>& f, const device half* r0, const device half* r1) {
    const half4 a = *reinterpret_cast<const device half4*>(r0);
    const half4 b = *reinterpret_cast<const device half4*>(r1);
    f[0] = a.x; f[1] = a.y; f[2] = a.z; f[3] = a.w; f[4] = b.x; f[5] = b.y; f[6] = b.z; f[7] = b.w;
}
// H16: tile_matmad_nax's multiply-accumulates for the first 16-row fragment row only (fragment row 0 of Dt, both 16-deep
// slices in order): rows 0..15 take the same 16x32x16 MMAs in the same order as the full call, rows 16..31 none
inline void exl3pf_mma_top(thread NAXTile<float, 2, 2>& Dt, thread NAXTile<half, 2, 2>& At, thread NAXTile<half, 2, 2>& Bt) {
    STEEL_PRAGMA_UNROLL
    for (short kk = 0; kk < 2; ++kk) BaseNAXFrag::mma(Dt.frag_at(0, 0), Dt.frag_at(0, 1), At.frag_at(0, kk), metal::bool_constant<false>{}, Bt.frag_at(0, kk), Bt.frag_at(1, kk), metal::bool_constant<true>{});
}
// codeword m of a lane: the general reader (lp = its bit offset), or (IR, n % 16 == 0) the integer-rate one (lp = its b:
// one select only where window m can carry, one extract); both read the same IntWords, so the same bits
template <uint N, uint WMASK, uint M, int IR> inline uint exl3pf_cw(thread const Exl3BitsM<exl3_rmode<N>()>& w, uint lp) {
    return exl3_tile_cw<N, WMASK, M, IR>(w, lp);
}
template <uint N, uint WMASK, int CB, int DF, int IR>
inline void exl3pf_bfrag(thread vec<half, 8>& f, const device uint* tw, uint qb, uint ga, uint lp) {
    const auto ww = exl3_tile_load<N, IR>(tw, qb, ga, lp);
    f[0] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 0, IR>(ww, lp));
    f[1] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 1, IR>(ww, lp));
    f[2] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 2, IR>(ww, lp));
    f[3] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 3, IR>(ww, lp));
    f[4] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 4, IR>(ww, lp));
    f[5] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 5, IR>(ww, lp));
    f[6] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 6, IR>(ww, lp));
    f[7] = exl3pf_w<CB, DF>(exl3pf_cw<N, WMASK, 7, IR>(ww, lp));
}
