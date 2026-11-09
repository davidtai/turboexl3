} else {
  // The predictor's threadgroups, as router/pred_open's, after the rows launch's NRG * (NE / 16) router threadgroups
  const uint ptg = tg - uint(NRG * (NE / 16));
