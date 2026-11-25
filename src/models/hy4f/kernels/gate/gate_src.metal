        // o at the element, in the layout O_FLAT names: 0, o's [1, H, S, VH] (head o / VH, row `row`, column o % VH: the
        // chain's transpose, read in place); 1, its rows' [1, S, H, VH] (element o of row `row`: kvb_mix's un-absorb
        // output as it lies, of which the host's o is the transposed view). A simdgroup's lanes read 32 consecutive
        // floats either way.
        const float ov = (O_FLAT != 0) ? o_src[ulong(row) * DIMC + o]
                                       : o_src[(ulong(o / VHC) * S + row) * VHC + o % VHC];
