        {
  constexpr float HAD_SCALE = 0.08838834764831845f;
  const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
  const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
  for (uint q = 0; q < 4u; ++q) {
    const uint o = s_b * 128u + s_L + 32u * q;
    float acc = s_zgc[ulong(s_row) * ulong(S_INT) + o];
    for (uint k = 1u; k < uint(SKSG); ++k) acc += s_zgc[(ulong(k) * ulong(uint(SRS)) + s_row) * ulong(S_INT) + o];
    if (uint(SDEF) != 0u) acc = fma(acc, MUL1_A, MUL1_C * s_xg);
    g[q] = float(half(acc));
  }
  turboexl3_h128r(g, s_L);
  for (uint q = 0; q < 4u; ++q) g[q] = float(half(g[q] * HAD_SCALE * float(sh_gate_svh[s_b * 128u + s_L + 32u * q])));
        }
