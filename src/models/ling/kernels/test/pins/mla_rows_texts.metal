#pragma METAL fp contract
v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
inter[tm][tn] = static_cast<bfloat16_t>(mat[tm * IN + bn + tn]);
result[r][tm] += inter[tm][tn] * v_coeff[tn];
result[r][tm] += simd_shuffle_down(result[r][tm], sn);
tgp_memory[(r * BN + sgN) * (blockM + TM) + bm + tm] = result[r][tm];
result[r][tm] += tgp_memory[(r * BN + sgn) * (blockM + TM) + bm + tm];
y[size_t(r0 + r) * size_t(OUT) + size_t(out_row + tm)] = static_cast<bfloat16_t>(result[r][tm]);
