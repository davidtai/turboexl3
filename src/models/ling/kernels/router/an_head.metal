// The post-attention add_norm's normed rows (an_open's sums), each word made where the router or the predictor reads x:
// the residual add rounded to bf16, then w * bf16(h * inv) rounded to bf16, mlx-serve's add_norm statements (the plain
// arm's delta, its store of HN)
struct turboexl3_ling_xn {
  const device bfloat16_t* h;
  const device bfloat16_t* xd;
  const device bfloat16_t* nw;
  float inv;
  bfloat16_t operator[](int c) const {
    const bfloat16_t hn = bfloat16_t(float(h[c]) + float(xd[c]));
    return bfloat16_t(float(nw[c]) * float(bfloat16_t(float(hn) * inv)));
  }
};
// The same rows read as 4-vectors, x4's layout: element j is row j / ld4's columns 4 * (j % ld4) to 4 * (j % ld4) + 3
struct turboexl3_ling_xn4 {
  const device bfloat16_t* h;
  const device bfloat16_t* xd;
  const device bfloat16_t* nw;
  const threadgroup float* inv;
  int ld4;
  vec<bfloat16_t, 4> operator[](int j) const {
    const int r = j / ld4;
    const int c = (j % ld4) * 4;
    vec<bfloat16_t, 4> o;
    for (int i = 0; i < 4; i++) {
      const size_t at = size_t(r) * size_t(ld4) * 4 + size_t(c + i);
      const bfloat16_t hn = bfloat16_t(float(h[at]) + float(xd[at]));
      o[i] = bfloat16_t(float(nw[c + i]) * float(bfloat16_t(float(hn) * inv[r])));
    }
    return o;
  }
};
