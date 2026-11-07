if (tg < uint(R * (NE / 16))) {
  const uint row = tg / uint(NE / 16);
  const uint blk = tg % uint(NE / 16);
