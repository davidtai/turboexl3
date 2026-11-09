        const T hn = T(float(H[at]) + delta);
        H_NEW[at] = hn;
        hv[p * 4 + i] = float(hn);
    }
}
// the planted fault (tail_test.zig's red check, never served): the thread's squares summed in reverse column order,
// a pure reordering of the add_norm's sum of squares
for (int q = NC * 4 - 1; q >= 0; q--) acc += hv[q] * hv[q];
acc = simd_sum(acc);
if (sg == 0) local_sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) local_sums[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float total = simd_sum(local_sums[lane]);
const float inv = metal::precise::rsqrt(total / float(D) + eps);
for (int p = 0; p < NC; p++)
for (int i = 0; i < 4; i++) {
    const int c = p * TN * 4 + int(t) * 4 + i;
    if (c < D) HN[size_t(r) * D + c] = T(float(W[c]) * float(T(hv[p * 4 + i] * inv)));
}