    float vg[4]; float vu[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; const float xr = float(half(float(x[row * DIMC + i]))); vg[j] = xr * float(gate_suh[e * DIMC + i]); vu[j] = xr * float(up_suh[e * DIMC + i]); }
    for (uint s = 0; s < 5u; ++s) { const uint bit = 1u << s; for (uint j = 0; j < 4u; ++j) { const float pg = simd_shuffle_xor(vg[j], ushort(bit)); const float pu = simd_shuffle_xor(vu[j], ushort(bit)); vg[j] = (L & bit) ? (pg - vg[j]) : (vg[j] + pg); vu[j] = (L & bit) ? (pu - vu[j]) : (vu[j] + pu); } }
    { float t0 = vg[0] + vg[1], t1 = vg[0] - vg[1], t2 = vg[2] + vg[3], t3 = vg[2] - vg[3]; vg[0] = t0 + t2; vg[1] = t1 + t3; vg[2] = t0 - t2; vg[3] = t1 - t3; }
    { float t0 = vu[0] + vu[1], t1 = vu[0] - vu[1], t2 = vu[2] + vu[3], t3 = vu[2] - vu[3]; vu[0] = t0 + t2; vu[1] = t1 + t3; vu[2] = t0 - t2; vu[3] = t1 - t3; }
    float sgv = 0.0f, suv = 0.0f;
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint r = i & 15u; const uint q = (r & 7u) >> 1u; const uint m = (r & 1u) | ((r >> 3u) << 1u);
        const ulong o = ulong(rk) * ulong(DIMC) + (i & ~15u) + 4u * q + m;
        const half hg = half(vg[j] * HAD_SCALE);
        const half hu = half(vu[j] * HAD_SCALE);
        xg[o] = hg;
        xu[o] = hu;
        sgv += float(hg);
        suv += float(hu);
    }
    // each 128-block's sum of the rounded inputs (the deferred MUL1 map needs the row-slot's input sum)
    sgv = simd_sum(sgv);
    suv = simd_sum(suv);
    if (L == 0u) { sg[ulong(rk) * ulong(DIMC / 128u) + b] = sgv; su[ulong(rk) * ulong(DIMC / 128u) + b] = suv; }
