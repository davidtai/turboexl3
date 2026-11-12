    // Launch 1, the query absorbed: ql[s, h, c] = sum over d < DN of qn[h, s, d] * w[h (DN + DV) + d, c], in f32 (each
    // operand converted exactly, one fma a term, d ascending, from 0), for c < DL. One thread: 4 columns of one head, every
    // row s in turn. qn [1, H, S, DN] (row-contiguous), w kv_b [H (DN + DV), DL] (row h (DN + DV) + d: the key half's row
    // d of head h), ql f32 [S, H, DL].
    const uint gx = thread_position_in_grid.x;
    const uint h = gx / (uint(DL) / 4u);
    const uint c0 = (gx % (uint(DL) / 4u)) * 4u;
    if (h >= uint(H)) return;
    const device WT* wh = w + ulong(h) * ulong(DN + DV) * ulong(DL) + ulong(c0);
    for (uint s = 0; s < uint(S); ++s) {
        const device QT* qrow = qn + (ulong(h) * ulong(S) + ulong(s)) * ulong(DN);
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint d = 0; d < uint(DN); ++d) {
            const float qv = float(qrow[d]);
            const device WT* wr = wh + ulong(d) * ulong(DL);
            a0 = fma(qv, float(wr[0]), a0);
            a1 = fma(qv, float(wr[1]), a1);
            a2 = fma(qv, float(wr[2]), a2);
            a3 = fma(qv, float(wr[3]), a3);
        }
        device float* o = ql + (ulong(s) * ulong(H) + ulong(h)) * ulong(DL) + ulong(c0);
        o[0] = a0;
        o[1] = a1;
        o[2] = a2;
        o[3] = a3;
    }
