#pragma METAL fp contract(off)
v_coeff[r][tn] = static_cast<float>(in_vec[bn + tn]);
result[r][tm] += inter[tn] * v_coeff[r][tn];
result[r][tm] += simd_shuffle_down(result[r][tm], sn);
logits[(size_t)(row0 + r) * NE + out_row + tm] = static_cast<float>(result[r][tm]);
constexpr int RG = (R < 4) ? R : 4;
const uint ptg = tg - uint(NRG * (NE / 16));
v_coeff[tn] = static_cast<float>(in_vec[bn + tn]);
result[tm] += inter[tn] * v_coeff[tn];
result[tm] += simd_shuffle_down(result[tm], sn);
logits[(size_t)row * NE + out_row + tm] = static_cast<float>(result[tm]);
const int thrN = int(simd_lid);
const int simdM = int(simd_gid);
int bm = (simdM + thrM) * 4;
int bn = (simdN + thrN) * 4;
int out_row = int(blk) * 16 + bm;
out_row = out_row + 4 <= NE ? out_row : NE - 4;
auto mat = w + size_t(out_row) * HK;
for (int i = 0; i < HK / 128; ++i) {
int mat_offset = 0;
mat_offset += HK;
bn += 128;
for (ushort sn = 16; sn >= 1; sn >>= 1) {
if (simdN == 0 && thrN == 0) {
