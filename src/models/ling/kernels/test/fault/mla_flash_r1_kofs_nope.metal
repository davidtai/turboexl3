            // test fault (turboexl3_test_ling_mla_flash_r1_nope_fault): every 16-wide slice of a key's dims read from
            // the nope keys' row at k's stride, the rope dims past KN too (in the rope-once cache, the head's values that
            // follow its nope keys): the red check's wrong words
            const bool k_rope = false;
            const device T* K_src = k_rope ? KR : K;
            const int K_ld = k_rope ? int(kr_strides[2]) : int(k_strides[2]);
            const int K_load_off = ik * kU * K_ld + (k_rope ? id * kU - KN : id * kU);
