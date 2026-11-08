NAXTile<float, TM, TN> Dtile;
NAXTile<AccumType, TM, TN> Dtile;
NAXTile<float, RA, CA> Atile;
NAXTile<T, RA, CA> Atile;
NAXTile<float, RB, CB> Btile;
NAXTile<T, RB, CB> Btile;
      rp_plus_zero(Btile);
tile_matmad_nax(
#define kernel kernel, max_total_threads_per_threadgroup(WM * WN * 32)
#undef kernel
Dtile.store(D, int(ldd));
Dtile.store_safe(D, int(ldd), short2(sgp_sn, sgp_sm));
  constexpr short TM = SM / 16;
  return Dtile;
