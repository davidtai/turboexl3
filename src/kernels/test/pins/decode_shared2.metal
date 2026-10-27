const uint rk = threadgroup_position_in_grid.y - uint(SYOFF);
const uint rk = threadgroup_position_in_grid.y;
turboexl3_scw<N, WMASK, 7>(lo, mid, hi)
as_type<ulong>
