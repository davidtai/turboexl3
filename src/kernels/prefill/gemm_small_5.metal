    if (rows >= 32) Dt.store(zp, NDC);
    else Dt.store_safe(zp, NDC, short2(32, rows));
