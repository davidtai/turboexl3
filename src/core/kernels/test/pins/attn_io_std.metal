float v[4];
simd_shuffle_xor
float t0
const device half* z
half(
y[j] = z[ulong(row) * zs + zo + blk * 128u + L + 32u * j];
const uint hh = i / HDC;
const uint d = i - hh * HDC;
const T av = attn[(ulong(hh) * S + row) * HDC + d];
const T gv = gate[ulong(row) * uint(GROW) + hh * uint(GHEAD) + uint(GOFF) + d];
const T xv = av * sigtab[as_type<ushort>(gv)];
xg[ulong(row) * DIMC + i] = xv;
