    // the inner product's store (trunk_q8.zig's inner path, Config.pf_trunk_int8_inner): z half [M, N] of the tile's staged
    // halves, 8 a thread at a time (16 bytes), the rows past M never stored. The chain's finish reads it where it reads the
    // matmul's inner product (Dense.finishOf, finishPieces, ATTN-IN's and ATTN-OUT's finishes), so a linear's y is trunk
    // int8's (pg2's finish over these halves) wherever that finish runs text_finish's statements.
    for (uint i = (simd_group_id * 32u + thread_index_in_simdgroup) * 8u; i < uint(BM * BN); i += uint(WM * WN) * 256u) {
        const uint rt = i / uint(BN);
        if (c_row + int(rt) >= M) break;
        device half* zo = z + ulong(c_row + int(rt)) * ulong(N) + ulong(c_col) + ulong(i % uint(BN));
        *reinterpret_cast<device half4*>(zo) = *reinterpret_cast<const threadgroup half4*>(zt + i);
        *reinterpret_cast<device half4*>(zo + 4) = *reinterpret_cast<const threadgroup half4*>(zt + i + 4u);
    }
