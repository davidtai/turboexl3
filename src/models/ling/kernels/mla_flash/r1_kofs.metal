            // a 16-wide slice of the keys' dims lies whole in the nope keys (k) or in the roped key (kr), each at its own
            // row stride: the words the expanded cache's slice holds, into the same tile
            const bool k_rope = id * kU >= KN;
            const device T* K_src = k_rope ? KR : K;
            const int K_ld = k_rope ? int(kr_strides[2]) : int(k_strides[2]);
            const int K_load_off = ik * kU * K_ld + (k_rope ? id * kU - KN : id * kU);
