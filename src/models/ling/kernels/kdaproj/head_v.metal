    // the variant launch's head (ling/kdaproj.zig `kdaProjB` with a variant on): kdaproj/head's with member 0 at BT0
    // output tiles a threadgroup and members 1 and 2 at BT1 (Config.kda_bt0, kda_bt1; each member's column groups,
    // threadgroups and arrivals from its own), and with BF the members' threadgroups after b's OUTB / 4 (kdaproj/bfirst).
    // A threadgroup's member, K slice, block, rows and buffers are kdaproj/head's at its member's tiles; DEC-ONE's
    // statements then run under the member's BTC (kdaproj/bt_0, bt_1)
    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint INC = uint(IN);
    constexpr uint IN_TILES = INC / 16u;
    constexpr uint RBC = uint(RB);
    constexpr uint ROWSC = uint(ROWS);
    constexpr uint NCHC = (ROWSC + RBC - 1u) / RBC;   // row chunks
    constexpr uint NB = INC / 128u;                   // input 128-blocks
    constexpr uint GC0 = 8u / uint(BT0);              // column groups per 128-wide output block: member 0's
    constexpr uint GC1 = 8u / uint(BT1);              // members 1 and 2's
    constexpr uint BTX = (uint(BT0) > uint(BT1)) ? uint(BT0) : uint(BT1);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    // each member's threadgroups per K slice, where its threadgroups start, and b's before the members' (BF)
    constexpr uint NT0 = uint(OUT0) / 128u * GC0;
    constexpr uint NT1 = uint(OUT1) / 128u * GC1;
    constexpr uint NT2 = uint(OUT2) / 128u * GC1;
    constexpr uint F1 = NT0 * uint(KS0);
    constexpr uint F2 = F1 + NT1 * uint(KS1);
    constexpr uint NBF = (BF != 0) ? uint(OUTB) / 4u : 0u;
    const uint flat = threadgroup_position_in_grid.x - NBF;
    const uint jm = (flat < F1) ? 0u : ((flat < F2) ? 1u : 2u);
    const uint mflat = flat - ((jm == 0u) ? 0u : ((jm == 1u) ? F1 : F2));
    const uint mnt = (jm == 0u) ? NT0 : ((jm == 1u) ? NT1 : NT2);
    const uint OUTC = (jm == 0u) ? uint(OUT0) : ((jm == 1u) ? uint(OUT1) : uint(OUT2));
    const uint OUT_TILES = OUTC / 16u;
    const uint KSC = (jm == 0u) ? uint(KS0) : uint(KS1);
    const uint GCM = (jm == 0u) ? GC0 : GC1;          // the member's column groups
    const uint ARR = GCM * KSC;                       // arrivals per (output block, row chunk)
    const uint PER = IN_TILES / (KSC * 4u);
    const uint blk = mflat % mnt;
    const uint chunk = threadgroup_position_in_grid.y;
    const uint r0 = chunk * RBC;
    const uint ksl = mflat / mnt;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint ob = blk / GCM;
    threadgroup float red[4][RBC * 16u * BTX];
    threadgroup uint arrived_last;
