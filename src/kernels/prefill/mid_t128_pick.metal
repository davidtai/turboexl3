    const bool qt = int(p) < xbase[e];
    const device half* zgs = qt ? zgt : zg;
    const device half* zus = qt ? zut : zu;
