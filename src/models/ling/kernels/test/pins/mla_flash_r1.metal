const device T* KR = kr + tidl.z * kr_strides[0]; // Batch
const bool k_rope = id * kU >= KN;
const device T* K_src = k_rope ? KR : K;
const int K_ld = k_rope ? int(kr_strides[2]) : int(k_strides[2]);
const int K_load_off = ik * kU * K_ld + (k_rope ? id * kU - KN : id * kU);
KR += BK * int(kr_strides[2]);
K_src + K_load_off
K + K_load_off
const int K_load_off = ik * kU * int(k_strides[2]) + id * kU;
K += BK * int(k_strides[2]);
const bool k_rope = false;
