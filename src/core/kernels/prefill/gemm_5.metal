            if (full) Dt.store(zp, NDC);
            else Dt.store_safe(zp, NDC, short2(SN, sgp));
        }
        if (cb + 1 < NCBC) Dt.clear();
    }
