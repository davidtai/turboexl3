    // the member's trellis (its own buffer: base 0), its own count of output tiles and the block's tile counted from its
    // first (`jtile`), and its svh read at the pair's output column (member 1's columns start at OB1 128-blocks)
    const auto w = (jm == 0u) ? w0 : w1;
    struct LingPairSvh {
        const device half* p0;
        const device half* p1;
        uint o1;
        half operator[](uint o) const { return (o < o1) ? p0[o] : p1[o - o1]; }
    };
    const LingPairSvh svh = { svh0, svh1, uint(OB1) * 128u };
    const uint mo0 = (jm == 0u) ? 0u : uint(OB1) * 8u;
    const uint mtiles = (jm == 0u) ? uint(OB1) * 8u : OUT_TILES - uint(OB1) * 8u;
    const ulong mbase = 0ul;
