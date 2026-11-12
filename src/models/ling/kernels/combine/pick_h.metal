// JOINLESS's source choice: source s of the 24 a launch binds (an unused slot aliases source 0; s 23 and above the last)
inline const device half* ling_combine_pick_h(int s, const device half* s0, const device half* s1, const device half* s2, const device half* s3, const device half* s4, const device half* s5, const device half* s6, const device half* s7, const device half* s8, const device half* s9, const device half* s10, const device half* s11, const device half* s12, const device half* s13, const device half* s14, const device half* s15, const device half* s16, const device half* s17, const device half* s18, const device half* s19, const device half* s20, const device half* s21, const device half* s22, const device half* s23) {
    switch (s) {
        case 0: return s0;
        case 1: return s1;
        case 2: return s2;
        case 3: return s3;
        case 4: return s4;
        case 5: return s5;
        case 6: return s6;
        case 7: return s7;
        case 8: return s8;
        case 9: return s9;
        case 10: return s10;
        case 11: return s11;
        case 12: return s12;
        case 13: return s13;
        case 14: return s14;
        case 15: return s15;
        case 16: return s16;
        case 17: return s17;
        case 18: return s18;
        case 19: return s19;
        case 20: return s20;
        case 21: return s21;
        case 22: return s22;
        default: return s23;
    }
}
