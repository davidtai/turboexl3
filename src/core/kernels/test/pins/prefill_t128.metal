const_cast
const bool qt = int(p) < xbase[e];
const device half* zgs = qt ? zgt : zg;
const device half* zus = qt ? zut : zu;
g[j] = float(zgs[ulong(p) * IC + i]);
u[j] = float(zus[ulong(p) * IC + i]);
float(zg[
float(zu[
float(zgs[
float(zus[
const device half* zds = (int(p) < xbase[e]) ? zdt : zd;
v[j] = float(zds[ulong(p) * HC
float(zd[
float(zds[
const uint NPQ = min(EC, SC / (CU + 1u));
const uint TQ = (T128 != 0) ? max(1u, NPQ + (SC - (CU + 1u) * NPQ) / 128u) : 1u;
constexpr uint PER = CU / BMC + ((CU % BMC > SR) ? 1u : 0u);
constexpr uint MINR = (BMC < SR + 1u) ? BMC : SR + 1u;
const uint TBIG = (T128 != 0) ? max(1u, min(TBIG0, min(EC * PER, SC / MINR))) : TBIG0;
constexpr uint CU = (uint(CUT) < 127u) ? uint(CUT) : 127u;
if (T128 != 0 && cnt > CU) { nq += 1u; cnt = 0u; }
if (r == 0u) nb = nfull;
else if (r > SR) nb = nfull + 1u;
else { nb = nfull; ns = (r + 31u) / 32u; }
