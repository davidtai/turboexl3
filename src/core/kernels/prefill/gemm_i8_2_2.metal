            if (m < rows) z[ulong(row0 + m) * ulong(NDC) + ulong(n0 + tn + int(ids[0]))] = half(fma(v, 4.0f * MA, (510.0f * MA + MC) * xsum[m]));
