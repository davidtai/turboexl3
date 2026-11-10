#include <metal_tensor>
// int8 prefill (prefill.Options.int8): a MUL1 codeword's weight as int8, round((t - 510) / 4) of its byte sum t (the sum
// exl3_mul1_t computes, 0 .. 1005 over every 16-bit codeword, so the clamp never acts); the GEMM's epilogue applies 4 A and
// 510 A + C
inline int8_t exl3pf_q8(uint cw) {
    const uint rr = cw * 0x83DCD12Du;
    const uint bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
    const int t = int((bs + (bs >> 16u)) & 0xFFFFu);
    return int8_t(clamp((t - 508) >> 2, -128, 127));
}
// one 128-deep K block of a 64-wide output block's trellis as int8 B [n][k] (threadgroup memory, k fastest, 128 bytes a
// row): simdgroup sg decodes the KTS = 32 / NSG k tiles of n tile sg % 4 from k tile (sg / 4) KTS on. Lane L (c = L >> 2,
// q = L & 3) holds a tile's weights at n c, c + 8 and k 2q, 2q + 1 (pair 0), 2q + 8, 2q + 9 (pair 1), exl3pf_stage's
// places, and takes its k tiles in a rotated order (step i: kbase + (c + i) % KTS), so a store's 32 lanes write 16
// distinct banks: eight k tiles at KTS 8; at KTS 4 lanes c >= 4 store pair 1 first
template <uint N, uint WMASK, int IR, int N16, int NSG>
inline void exl3pf_stage_q8(const device uint* wbk, threadgroup int8_t* bq, uint sg, uint qb, uint ga, uint lp, uint L) {
    constexpr uint KTS = 32u / uint(NSG);
    const uint c = L >> 2u;
    const uint q = L & 3u;
    const uint nt = sg & 3u;
    const uint kbase = (sg >> 2u) * KTS;
    const bool sw = KTS < 8u && c >= KTS;
    const uint o0 = sw ? 8u : 0u;
    for (uint i = 0; i < KTS; ++i) {
        const uint kt = kbase + ((c + i) % KTS);
        const auto ww = exl3_tile_load<N, IR>(wbk + (ulong(kt) * ulong(N16) + ulong(nt)) * ulong(N / 2u), qb, ga, lp);
        const char2 p0 = char2(exl3pf_q8(exl3pf_cw<N, WMASK, 0, IR>(ww, lp)), exl3pf_q8(exl3pf_cw<N, WMASK, 1, IR>(ww, lp)));
        const char2 p1 = char2(exl3pf_q8(exl3pf_cw<N, WMASK, 2, IR>(ww, lp)), exl3pf_q8(exl3pf_cw<N, WMASK, 3, IR>(ww, lp)));
        const char2 p2 = char2(exl3pf_q8(exl3pf_cw<N, WMASK, 4, IR>(ww, lp)), exl3pf_q8(exl3pf_cw<N, WMASK, 5, IR>(ww, lp)));
        const char2 p3 = char2(exl3pf_q8(exl3pf_cw<N, WMASK, 6, IR>(ww, lp)), exl3pf_q8(exl3pf_cw<N, WMASK, 7, IR>(ww, lp)));
        threadgroup int8_t* d = bq + (nt * 16u + c) * 128u + kt * 16u + 2u * q;
        *reinterpret_cast<threadgroup char2*>(d + o0) = sw ? p1 : p0;
        *reinterpret_cast<threadgroup char2*>(d + (8u - o0)) = sw ? p0 : p1;
        *reinterpret_cast<threadgroup char2*>(d + 1024u + o0) = sw ? p3 : p2;
        *reinterpret_cast<threadgroup char2*>(d + 1024u + (8u - o0)) = sw ? p2 : p3;
    }
}
