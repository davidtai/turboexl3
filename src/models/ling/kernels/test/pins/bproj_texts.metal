v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
inter[tn] = static_cast<bfloat16_t>(mat[mat_offset + bn + tn]);
result[tm] += inter[tn] * v_coeff[tn];
result[tm] += simd_shuffle_down(result[tm], sn);
total[tm] = (sgN == 0) ? result[tm] : (total[tm] + result[tm]);
