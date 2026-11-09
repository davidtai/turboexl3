    // turboexl3_ling_mla_flash_r1, the cache with the roped key once a token: k holds each head's nope keys (a key's
    // first KN dims, at k's strides), kr the token's one roped key (its last BD - KN dims), shared by every head, so
    // kr has a batch offset and no head offset
    const device T* KR = kr + tidl.z * kr_strides[0]; // Batch
