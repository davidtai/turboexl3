// turboexl3 on HIP: the lane reader and the codebooks alone, for src/hip_tests.zig (no product kernel launches these).

#include "lane.hpp"

// Grid (tiles), 32 threads: each lane's eight codewords of each tile at rate n and codeword mask `mask`, in code-position
// order (8 lane + m), and their MCG and MUL1 values (fp16 bits). Trellis: tiles of n halfwords, back to back.
extern "C" __global__ void __launch_bounds__(32) t3_test_lane(const uint32_t* trellis, uint32_t n, uint32_t mask,
                                                              uint16_t* cw_out, uint16_t* mcg_out, uint16_t* mul1_out) {
    const uint32_t tile = blockIdx.x, lane = threadIdx.x;
    uint32_t cw[8];
    t3::lane_codewords(trellis + static_cast<size_t>(tile) * (n / 2u), n, lane, mask, cw);
#pragma unroll
    for (uint32_t m = 0; m < 8u; ++m) {
        const size_t at = static_cast<size_t>(tile) * 256u + 8u * lane + m;
        cw_out[at] = static_cast<uint16_t>(cw[m]);
        mcg_out[at] = t3::decode<t3::kMcg>(cw[m]);
        mul1_out[at] = t3::decode<t3::kMul1>(cw[m]);
    }
}

// Grid (256), 256 threads: every 16-bit codeword's MCG and MUL1 values and MUL1's byte sum.
extern "C" __global__ void __launch_bounds__(256) t3_test_codebooks(uint16_t* mcg_out, uint16_t* mul1_out,
                                                                    uint16_t* t_out) {
    const uint32_t cw = blockIdx.x * 256u + threadIdx.x;
    mcg_out[cw] = t3::decode_mcg(cw);
    mul1_out[cw] = t3::decode_mul1(cw);
    t_out[cw] = static_cast<uint16_t>(t3::mul1_t(cw));
}
