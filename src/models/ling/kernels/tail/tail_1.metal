// Ling's MoE tail (tail.zig): mlx-serve's add_norm (sdk/blocks/norms/add_norm.zig, its single-row kernel: TN threads
// of four columns a row, one threadgroup a row) with the add's delta made in place of `delta = float(X[at]);`
const uint t = thread_position_in_threadgroup.x;
const uint r = threadgroup_position_in_grid.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
constexpr int NC = (D + TN * 4 - 1) / (TN * 4);
threadgroup float local_sums[32];
float hv[NC * 4];
float acc = 0.0f;
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * TN * 4 + int(t) * 4 + i;
    hv[p * 4 + i] = 0.0f;
    if (c < D) {
        const size_t at = size_t(r) * D + c;
        float delta;
