// The decode join's source choice (moeTailParts): source s of the 8 a launch binds (an unused slot aliases source 0)
inline const device float* ling_tail_pick(int s, const device float* s0, const device float* s1, const device float* s2, const device float* s3, const device float* s4, const device float* s5, const device float* s6, const device float* s7) {
    switch (s) {
        case 0: return s0;
        case 1: return s1;
        case 2: return s2;
        case 3: return s3;
        case 4: return s4;
        case 5: return s5;
        case 6: return s6;
        default: return s7;
    }
}
