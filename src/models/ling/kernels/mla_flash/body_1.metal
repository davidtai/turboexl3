    // turboexl3_ling_mla_flash: MLX 0.32.5's attention_nax (steel/attn/kernels/steel_attention_nax.h, Copyright © 2024-26
    // Apple Inc., MIT), its statements in its order, less its mask, sinks, log-sum-exp and window arms (none taken here).
    // Its AttnParams come from the operands (q [B, H, qL, BD], k [B, Hk, kL, BD], v [B, Hk, kL, BV] at their own strides,
    // o contiguous) and `scale`; its function constants align_Q, align_K and do_causal are the template constants
    // ALIGN_Q, ALIGN_K and CAUSAL. Causal from the bottom right as MLX aligns it (query row i sees keys up to kL - qL + i).
    // Launched as MLX launches it: threadgroups (ceil(qL / BQ), H, B) of (32, WM, WN) threads.
    constexpr bool align_Q = ALIGN_Q != 0;
    constexpr bool align_K = ALIGN_K != 0;
    constexpr bool do_causal = CAUSAL != 0;

    const int qL = int(q_shape[2]);
    const int kL = int(k_shape[2]);
    const int gqa_factor = int(q_shape[1]) / int(k_shape[1]);
    const int NK = (kL + BK - 1) / BK;
    const int NQ_aligned = qL / BQ;
    const int NK_aligned = kL / BK;
    const int qL_rem = qL - NQ_aligned * BQ;
    const int kL_rem = kL - NK_aligned * BK;
    const int qL_off = kL - qL;
    const int64_t O_strides[3] = {int64_t(q_shape[1]) * int64_t(qL) * int64_t(BV), int64_t(qL) * int64_t(BV), int64_t(BV)};

    const uint simd_group_id = simdgroup_index_in_threadgroup;
    const uint3 tid = threadgroup_position_in_grid;

    // Move to correct block
    ulong3 tidl{tid.x, tid.y, tid.z};

    const device T* Q = q;
    const device T* K = k;
    const device T* V = v;
    device T* O = o;

    Q += tidl.z * q_strides[0] + // Batch
        tidl.y * q_strides[1] + // Head
        tidl.x * BQ * q_strides[2]; // Sequence

    ulong kv_head_idx = int(tid.y) / gqa_factor;
    K += tidl.z * k_strides[0] + // Batch
        kv_head_idx * k_strides[1]; // Head
