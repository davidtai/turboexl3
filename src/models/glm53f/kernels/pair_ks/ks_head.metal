    // dense's dec_head for two members of one input at their own K splits (member 0's OB1 128-blocks of output at KS0,
    // member 1's OUT / 128 - OB1 at KS1): the threadgroups lie flat along x, member 0's (its output blocks' column groups,
    // slice by slice) then member 1's; each takes its member's slice (ksl), slice count (KSC), tiles a slice and simdgroup
    // (PER), inputs a slice (MSPAN, SPB 128-blocks) and arrivals (ARR). The prepare's buffer keeps the wider slice's inputs
    // as its row stride (SPAN).
    constexpr uint N = uint(NHW);
    constexpr uint NW = N / 2u;
    constexpr uint WMASK = (WBITS >= 16) ? 0xffffu : ((1u << uint(WBITS)) - 1u);
    constexpr int CB = int(CODEBOOK);
    constexpr uint INC = uint(IN);
    constexpr uint OUTC = uint(OUT);
    constexpr uint IN_TILES = INC / 16u;
    constexpr uint OUT_TILES = OUTC / 16u;
    constexpr uint BTC = uint(BT);
    constexpr uint RBC = uint(RB);
    constexpr uint ROWSC = uint(ROWS);
    constexpr uint NCHC = (ROWSC + RBC - 1u) / RBC;   // row chunks
    constexpr uint NB = INC / 128u;                   // input 128-blocks
    constexpr uint GC = 8u / BTC;                     // column groups per 128-wide output block
    constexpr uint KS0C = uint(KS0);
    constexpr uint KS1C = uint(KS1);
    constexpr uint PER0 = IN_TILES / (KS0C * 4u);
    constexpr uint PER1 = IN_TILES / (KS1C * 4u);
    constexpr uint SPAN = 64u * ((PER0 > PER1) ? PER0 : PER1);
    constexpr uint NG0 = uint(OB1) * GC;              // member 0's threadgroups a slice
    constexpr uint NG1 = (OUTC / 128u - uint(OB1)) * GC;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint fx = threadgroup_position_in_grid.x;
    const bool second = fx >= NG0 * KS0C;
    const uint fm = second ? fx - NG0 * KS0C : fx;
    const uint ngm = second ? NG1 : NG0;
    const uint ksl = fm / ngm;
    const uint blk = (second ? NG0 : 0u) + (fm - ksl * ngm);
    const uint KSC = second ? KS1C : KS0C;
    const uint PER = second ? PER1 : PER0;
    const uint MSPAN = 64u * PER;
    const uint SPB = MSPAN / 128u;
    const uint ARR = GC * KSC;                        // arrivals per (output block, row chunk)
    const uint chunk = threadgroup_position_in_grid.y;
    const uint r0 = chunk * RBC;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    const uint ob = blk / GC;
    threadgroup float red[4][RBC * 16u * BTC];
    threadgroup uint arrived_last;
    device TURBOEXL3_COH float* zc = (device TURBOEXL3_COH float*)z;
