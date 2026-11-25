        // the planted fault (a red check, never served): o read in the other layout than O_FLAT names (the rows' order
        // on a [1, H, S, VH] o, the heads' on a [1, S, H, VH] one)
        const float ov = (O_FLAT == 0) ? o_src[ulong(row) * DIMC + o]
                                       : o_src[(ulong(o / VHC) * S + row) * VHC + o % VHC];
