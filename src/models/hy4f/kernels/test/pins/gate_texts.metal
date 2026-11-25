OUT_T(float(half(v[j] * HAD_SCALE * float(svh[o]))))
const uint o = b * 128u + L + 32u * j;
const float ye = 1 / (1 + metal::precise::exp(metal::abs(yv)));
const float sg = (yv < 0) ? ye : 1 - ye;
const float ov = (O_FLAT != 0) ? o_src[ulong(row) * DIMC + o]
gated[ulong(row) * DIMC + o] = G_T(ov * sg);
: o_src[(ulong(o / VHC) * S + row) * VHC + o % VHC];
