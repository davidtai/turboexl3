        // a block value of the prepared rows xp (natural order: text_prep's half(v * HAD_SCALE) at this place)
        h[k][j] = xp[ulong(row) * DIMC + b * 128u + L + 32u * j];
