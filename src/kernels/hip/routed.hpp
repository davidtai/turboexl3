// turboexl3 on HIP: what the routed-expert groups share (decode and prefill): the 128-point Hadamard of one value a
// thread, the GEMV of a wave over its input tiles, and the finishes of the two rounding profiles.
#pragma once

#include "common.hpp"

namespace t3 {

// Expert e's trellis: the e-th matrix of a contiguous bank [E][K/16][N/16][n/2 words] (ptrs 0), or the address in
// entry e of a pointer table, u64 [E] (ptrs 1: a host whose loader keeps each expert's tensor apart).
__device__ __forceinline__ const uint32_t* expert_trellis(const uint32_t* bank, size_t e, size_t words, int ptrs) {
    return ptrs ? reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint64_t*>(bank)[e]) : bank + e * words;
}

// The unscaled Hadamard of a 128-block held one value a thread (tid 0..127): five lane stages, then two through
// xch (turboexl3_h128 of gemv_core.metal).
__device__ __forceinline__ float h128_thread(float v, float* xch, uint32_t tid) {
    const uint32_t lane = tid & 31u;
#pragma unroll
    for (uint32_t s = 0; s < 5u; ++s) {
        const uint32_t bit = 1u << s;
        const float p = __shfl_xor(v, static_cast<int>(bit), 32);
        v = (lane & bit) ? (p - v) : (v + p);
    }
#pragma unroll
    for (uint32_t s = 5; s < 7u; ++s) {
        const uint32_t bit = 1u << s;
        xch[tid] = v;
        __syncthreads();
        const float p = xch[tid ^ bit];
        __syncthreads();
        v = (tid & bit) ? (p - v) : (v + p);
    }
    return v;
}

// One wave's GEMV over input tiles [tk0, tk1) of an expert matrix into a[BT][2] (turboexl3_gemv_core's order: input
// tiles ascending, weights m = 0..7, accumulator m >> 2, input element m & 3 of the quad's lane-order four).
template <int CB, int DEFER, uint32_t BT>
__device__ __forceinline__ void gemv_tiles(const uint32_t* wbase, const uint16_t* xrow, uint32_t tk0, uint32_t tk1,
                                           uint32_t blk, uint32_t out_tiles, uint32_t n, uint32_t mask, uint32_t lane,
                                           float (&a)[BT][2]) {
    const uint32_t nw = n / 2u;
    const uint32_t base = lane_base(n, lane);
    uint32_t off[8];
#pragma unroll
    for (uint32_t m = 0; m < 8u; ++m) off[m] = lane_off(n, lane, m, base);
#pragma unroll
    for (uint32_t t = 0; t < BT; ++t) a[t][0] = a[t][1] = 0.0f;
    for (uint32_t tk = tk0; tk < tk1; ++tk) {
        float xr[4];
#pragma unroll
        for (uint32_t j = 0; j < 4u; ++j) xr[j] = half_value(xrow[tk * 16u + j]);
#pragma unroll
        for (uint32_t t = 0; t < BT; ++t) {
            const LaneWords ww = lane_words(wbase + (static_cast<size_t>(tk) * out_tiles + blk * BT + t) * nw, n, base);
#pragma unroll
            for (uint32_t m = 0; m < 8u; ++m) {
                const float w = weight<CB, DEFER>(lane_cw(ww, off[m], mask));
                a[t][m >> 2] = __builtin_fmaf(xr[m & 3u], w, a[t][m >> 2]);
            }
        }
    }
}

// The gate and up finish of one routed pair (z row `zr` of planes `planes` rows apart, ks of them in slice order), its
// SwiGLU (the profile's) and the down's prepare: output o = blk 128 + tid of the pair's hd row `hr` (lane order or
// natural), and the pair's 128-block sum into hs[hr][blk]. sgx / sux: the pair's input sums (the deferred map).
template <int PROF>
__device__ __forceinline__ void finish_gate_up(const float* zg, const float* zu, size_t zr, size_t planes, uint32_t ks,
                                               size_t e, const uint16_t* gate_svh, const uint16_t* up_svh,
                                               const uint16_t* down_suh, float sgx, float sux, uint16_t* hd, float* hs,
                                               size_t hr, uint32_t inter, uint32_t blk, int defer, float limit,
                                               bool lane_ord, float* xch, uint32_t tid) {
    const uint32_t lane = tid & 31u;
    const uint32_t o = blk * 128u + tid;
    float g = 0.0f, u = 0.0f;
    for (uint32_t k = 0; k < ks; ++k) {
        const size_t i = (static_cast<size_t>(k) * planes + zr) * inter + o;
        g += zg[i];
        u += zu[i];
    }
    if (defer) {
        g = __builtin_fmaf(g, mul1_a(), mul1_c() * sgx);
        u = __builtin_fmaf(u, mul1_a(), mul1_c() * sux);
    }
    float hv;
    if constexpr (PROF == 0) {
        g = h128_thread(round_half(g), xch, tid);
        u = h128_thread(round_half(u), xch, tid);
        const float gv = round_half(g * kHadScale * half_value(gate_svh[e * inter + o]));
        const float uv = round_half(u * kHadScale * half_value(up_svh[e * inter + o]));
        hv = round_half((gv / (1.0f + expf(-gv))) * uv);
    } else {
        g = h128_thread(g, xch, tid);
        u = h128_thread(u, xch, tid);
        float gv = bf16_value(bf16_bits(g * kHadScale * half_value(gate_svh[e * inter + o])));
        float uv = bf16_value(bf16_bits(u * kHadScale * half_value(up_svh[e * inter + o])));
        if (limit > 0.0f) {
            gv = gv > limit ? limit : gv;  // a NaN stays a NaN (torch.clamp)
            uv = uv > limit ? limit : (uv < -limit ? -limit : uv);
        }
        const float silu = bf16_value(bf16_bits(gv / (1.0f + expf(-gv))));
        hv = bf16_value(bf16_bits(silu * uv));
    }
    float d = hv * half_value(down_suh[e * inter + o]);
    d = h128_thread(d, xch, tid);
    const uint16_t hh = half_bits(d * kHadScale);
    hd[hr * inter + (lane_ord ? lane_order(o) : o)] = hh;
    float hsum = half_value(hh);
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) hsum += __shfl_xor(hsum, m, 32);
    __syncthreads();
    if (lane == 0u) xch[tid >> 5] = hsum;
    __syncthreads();
    if (tid == 0u) hs[hr * (inter / 128u) + blk] = ((xch[0] + xch[1]) + xch[2]) + xch[3];
    __syncthreads();
}

// The down finish of one routed pair at output o: its planes in slice order, the deferred map (hsum: the pair's input
// sum), then PROF 0: f16, H128, f16(v svh); PROF 1: H128, v svh in f32.
template <int PROF>
__device__ __forceinline__ float finish_down(const float* zd, size_t zr, size_t planes, uint32_t ks, float hsum,
                                             int defer, const uint16_t* down_svh, size_t e, uint32_t hidden, uint32_t o,
                                             float* xch, uint32_t tid) {
    float v = 0.0f;
    for (uint32_t q = 0; q < ks; ++q) v += zd[(static_cast<size_t>(q) * planes + zr) * hidden + o];
    if (defer) v = __builtin_fmaf(v, mul1_a(), mul1_c() * hsum);
    if constexpr (PROF == 0) {
        v = h128_thread(round_half(v), xch, tid);
        return round_half(v * kHadScale * half_value(down_svh[e * hidden + o]));
    } else {
        v = h128_thread(v, xch, tid);
        return v * kHadScale * half_value(down_svh[e * hidden + o]);
    }
}

}  // namespace t3
