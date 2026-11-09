    // pf (dense.dec_variants): DEC-ONE's GEMV loop with a rolling prefetch. Each input tile's words for lane L load a
    // pass ahead of their decode, into wc (the first tile's in the loop's first pass, beside the second's); the words,
    // the codewords, the decodes, the FMAs and their order are DEC-ONE's
    Exl3BitsM<exl3_rmode<N>()> wc[BTC];
