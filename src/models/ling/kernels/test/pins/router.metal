#pragma METAL fp contract(off)
result[tn] += vc * inter[tn];
result[tn] += simd_shuffle_down(result[tn], 4 * sm);
result[tm] += inter[tn] * v_coeff[tn];
scores[e] = turboexl3_ling_score(static_cast<float>(static_cast<bfloat16_t>(result[tm])), pb[e]);
constexpr int KL = (R + 4) / 5 == 1 ? 32 : 16;
acc += dot(wf[i], vec<float, 4>(x4[v * x_ld4 + base + i * KL + k_lane]));
scores[v * NE + out_row] = turboexl3_ling_score(static_cast<float>(static_cast<bfloat16_t>(result[v])), pb[out_row]);
return x * (metal::precise::log(xp1) / (xp1 - 1.0f));
s = (minval == -inf || maxval == inf) ? maxval : (maxval + turboexl3_ling_log1p(metal::precise::exp(minval - maxval)));
float q = metal::precise::sqrt(s);
auto y = 1 / (1 + metal::precise::exp(metal::abs(z)));
float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(l)));
scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
