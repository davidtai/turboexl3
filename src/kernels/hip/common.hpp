// turboexl3 on HIP (wave32): the helpers every kernel group shares.
//
// HIP forms of the Metal helpers in src/kernels/decode/gemv_core.metal (turboexl3_fold) and of the prepare and finish
// arithmetic in src/kernels/dense/prep_s.metal and dfinish_v.metal: the 128-point Hadamard of a lane's four values (index
// L + 32 j of a 128-block), the quad fold of a column's partial sums, and the activation loads and stores.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "lane.hpp"

namespace t3 {

// format.zig's activation numbering in the kernels: 0 f32, 1 f16, 2 bf16.
enum Dtype : int { kF32 = 0, kF16 = 1, kBf16 = 2 };

constexpr float kHadScale = 0.08838834764831845f;  // 1 / sqrt(128)

__device__ __forceinline__ float bf16_value(uint16_t b) { return __builtin_bit_cast(float, static_cast<uint32_t>(b) << 16); }

// f32 to bf16, nearest even (a NaN stays a NaN)
__device__ __forceinline__ uint16_t bf16_bits(float v) {
    uint32_t u = __builtin_bit_cast(uint32_t, v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

// element i of an activation row as f32
template <int DT>
__device__ __forceinline__ float load_act(const void* p, size_t i) {
    if constexpr (DT == kF32) return static_cast<const float*>(p)[i];
    else if constexpr (DT == kF16) return half_value(static_cast<const uint16_t*>(p)[i]);
    else return bf16_value(static_cast<const uint16_t*>(p)[i]);
}

// v stored in the output dtype (from an f16-rounded value, as the finish has it)
template <int DT>
__device__ __forceinline__ void store_act(void* p, size_t i, float v) {
    if constexpr (DT == kF32) static_cast<float*>(p)[i] = v;
    else if constexpr (DT == kF16) static_cast<uint16_t*>(p)[i] = half_bits(v);
    else static_cast<uint16_t*>(p)[i] = bf16_bits(v);
}

__device__ __forceinline__ float round_half(float v) { return half_value(half_bits(v)); }

// The unscaled Hadamard of a 128-block whose element L + 32 j lane L holds as v[j]: five lane stages, then the
// lane's own two stages (the order of prep_s.metal and dfinish_v.metal).
__device__ __forceinline__ void h128(float (&v)[4], uint32_t lane) {
#pragma unroll
    for (uint32_t s = 0; s < 5u; ++s) {
        const uint32_t bit = 1u << s;
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) {
            const float p = __shfl_xor(v[j], static_cast<int>(bit), 32);
            v[j] = (lane & bit) ? (p - v[j]) : (v[j] + p);
        }
    }
    const float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3];
    v[0] = t0 + t2;
    v[1] = t1 + t3;
    v[2] = t0 - t2;
    v[3] = t1 - t3;
}

// the lane-order slot of element i of a 128-block row (prep_s.metal LANE_ORDER): a quad's four rows of each tile in
// one 8-byte load
__device__ __forceinline__ uint32_t lane_order(uint32_t i) {
    const uint32_t r = i & 15u;
    const uint32_t q = (r & 7u) >> 1u;
    const uint32_t m = (r & 1u) | ((r >> 3u) << 1u);
    return (i & ~15u) + 4u * q + m;
}

// a column's sums over its quad, then each column's (lane L & 3 == 0) into red[t 16 + column] (turboexl3_fold)
template <uint32_t BT>
__device__ __forceinline__ void fold(float (&a)[BT][2], float* red, uint32_t lane) {
#pragma unroll
    for (uint32_t t = 0; t < BT; ++t)
#pragma unroll
        for (uint32_t c = 0; c < 2u; ++c) {
            a[t][c] += __shfl_xor(a[t][c], 1, 32);
            a[t][c] += __shfl_xor(a[t][c], 2, 32);
        }
    if ((lane & 3u) == 0u)
#pragma unroll
        for (uint32_t t = 0; t < BT; ++t) {
            red[t * 16u + (lane >> 2)] = a[t][0];
            red[t * 16u + (lane >> 2) + 8u] = a[t][1];
        }
}

// MUL1's deferred map: the decoded weight is A t + C, A = fp16 0x1EEE, C = 1024 A + fp16 0xC931 (both exact in f32)
__device__ __forceinline__ float mul1_a() { return half_value(0x1EEEu); }
__device__ __forceinline__ float mul1_c() { return 1024.0f * half_value(0x1EEEu) + half_value(0xC931u); }

// a weight as the inner product takes it: MUL1's t under the deferred map, else the decoded fp16 value
template <int CB, int DEFER>
__device__ __forceinline__ float weight(uint32_t cw) {
    if constexpr (DEFER != 0) return static_cast<float>(mul1_t(cw));
    else return half_value(decode<CB>(cw));
}

}  // namespace t3
