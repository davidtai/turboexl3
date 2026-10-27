// the merge's value in T (where the merge stores it), the gate's product in T, staged for the prepare
threadgroup T gst[BD];
const T xv = T(acc / Z);
const T gv = gate[ulong(s) * uint(GROW) + uint(hq) * uint(GHEAD) + uint(GOFF) + uint(d)];
gst[d] = xv * sigtab[as_type<ushort>(gv)];
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint HQN = threadgroups_per_grid.z;
const uint row = uint(s);
{
    constexpr uint DIMC = uint(DIM);
    constexpr uint NB = DIMC / 128u;
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    if (sgi < 2u) {
    const uint b = uint(hq) * 2u + sgi;
    float v[4];
    for (uint j = 0; j < 4u; ++j) { const uint i = b * 128u + L + 32u * j; v[j] = float(half(float(gst[sgi * 128u + L + 32u * j]))) * float(suh[i]); }
