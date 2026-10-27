uint row = thread_position_in_grid.y;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
const uint b = threadgroup_position_in_grid.x;
const uint sgi = simdgroup_index_in_threadgroup;
constexpr uint NSG = uint(TG) / 32u;

threadgroup float rk[NE];
threadgroup uint sel[NK];
threadgroup float tsm[2];
