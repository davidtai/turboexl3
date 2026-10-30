// turboexl3 on HIP (wave32, gfx11): the lane plan, the codeword reader and the two codebooks.
//
// The HIP form of src/kernels/lane/lane.metal's general reader (exl3_lane_base, exl3_lane_off, exl3_lane_words,
// exl3_lane_cw) and of src/format.zig's decoders (decodeMcg, decodeMul1). Lane L of a wave owns code positions
// 8L..8L+7 of a 256-weight tile (format.tensorCorePerm's order). Its eight windows end at stream bits
// ((8L + m + 1) n) >> 4 and lie in four consecutive words of the tile's circular bitstream, with the word before them for
// a window that starts in it. Every function takes the rate's halfwords n: a kernel passes a template constant (the
// integer rates) or the launch's value (any other even n), and the compiler folds the constant.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace t3 {

// format.zig Codebook's numbering.
enum Codebook : int { kMul1 = 0, kMcg = 1 };

struct LaneWords {
    uint32_t w0, w1, w2, w3, wprev;
};

// lane L's first needed word: (((8L + 1) n >> 4) + 16 n - 16) >> 5 (one period added keeps it positive)
__device__ __forceinline__ uint32_t lane_base(uint32_t n, uint32_t lane) {
    return ((((8u * lane + 1u) * n) >> 4u) + 16u * n - 16u) >> 5u;
}

// window m's end bit, counted from word `base`'s first bit (16 <= off <= 128)
__device__ __forceinline__ uint32_t lane_off(uint32_t n, uint32_t lane, uint32_t m, uint32_t base) {
    return (((8u * lane + m + 1u) * n) >> 4u) + 16u * n - 32u * base;
}

// the lane's four words from `base` and the word before them, modulo the tile's n / 2 words
__device__ __forceinline__ LaneWords lane_words(const uint32_t* tile, uint32_t n, uint32_t base) {
    const uint32_t nw = n / 2u;
    const uint32_t b = base % nw;
    LaneWords r;
    r.w0 = tile[b];
    r.w1 = tile[(b + 1u) % nw];
    r.w2 = tile[(b + 2u) % nw];
    r.w3 = tile[(b + 3u) % nw];
    r.wprev = tile[(b + nw - 1u) % nw];
    return r;
}

// the 16 bits ending at bit 0 of (lo : hi) >> sh
__device__ __forceinline__ uint32_t funnel(uint32_t lo, uint32_t hi, uint32_t sh) {
    return ((sh == 0u) ? hi : ((hi >> sh) | (lo << (32u - sh)))) & 0xffffu;
}

// the window ending `off` bits past the lane's first word, masked to the pack's codeword width
__device__ __forceinline__ uint32_t lane_cw(const LaneWords& w, uint32_t off, uint32_t mask) {
    const uint32_t q = (off - 1u) >> 5u;
    const uint32_t sh = 32u * (q + 1u) - off;
    const uint32_t hi = (q == 0u) ? w.w0 : ((q == 1u) ? w.w1 : ((q == 2u) ? w.w2 : w.w3));
    const uint32_t lo = (q == 0u) ? w.wprev : ((q == 1u) ? w.w0 : ((q == 2u) ? w.w1 : w.w2));
    return funnel(lo, hi, sh) & mask;
}

// the lane's eight codewords of one tile (`tile`: its first 32-bit word)
__device__ __forceinline__ void lane_codewords(const uint32_t* tile, uint32_t n, uint32_t lane, uint32_t mask,
                                               uint32_t (&cw)[8]) {
    const uint32_t base = lane_base(n, lane);
    const LaneWords w = lane_words(tile, n, base);
#pragma unroll
    for (uint32_t m = 0; m < 8u; ++m) cw[m] = lane_cw(w, lane_off(n, lane, m, base), mask);
}

__device__ __forceinline__ float half_value(uint32_t bits) {
    return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(bits)));
}

__device__ __forceinline__ uint16_t half_bits(float v) {
    return __builtin_bit_cast(uint16_t, static_cast<_Float16>(v));
}

// MCG: the hashed codeword's two fp16 halves summed in f32, rounded once to fp16 (format.decodeMcg)
__device__ __forceinline__ uint16_t decode_mcg(uint32_t cw) {
    const uint32_t pair = 0x3B603B60u ^ ((cw * 0xCBAC1FEDu) & 0x8FFF8FFFu);
    return half_bits(half_value(pair & 0xffffu) + half_value(pair >> 16u));
}

// MUL1's byte sum t in [0, 1020] of the hashed codeword
__device__ __forceinline__ uint32_t mul1_t(uint32_t cw) {
    const uint32_t rr = cw * 0x83DCD12Du;
    const uint32_t bs = (rr & 0x00FF00FFu) + ((rr >> 8u) & 0x00FF00FFu);
    return (bs + (bs >> 16u)) & 0xffffu;
}

// MUL1: fp16 1024 + t, then an f32 fused multiply-add by fp16 0x1EEE and 0xC931, rounded to fp16 (format.decodeMul1)
__device__ __forceinline__ uint16_t decode_mul1(uint32_t cw) {
    const float h = half_value(0x6400u + mul1_t(cw));
    return half_bits(__builtin_fmaf(h, half_value(0x1EEEu), half_value(0xC931u)));
}

template <int CB>
__device__ __forceinline__ uint16_t decode(uint32_t cw) {
    if constexpr (CB == kMcg) return decode_mcg(cw);
    else return decode_mul1(cw);
}

}  // namespace t3
