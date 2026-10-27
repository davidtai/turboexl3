m < 3u
suh0 : 
xp0 : 
for (uint m = 0; m < 4u; ++m) {
const device half* suh = (m == 0u) ? suh0 : ((m == 1u) ? suh1 : ((m == 2u) ? suh2 : suh3));
device half* xp = (m == 0u) ? xp0 : ((m == 1u) ? xp1 : ((m == 2u) ? xp2 : xp3));
xh[j] = float(half(float(x[row * DIMC + i])));
v[j] = xh[j] * float(suh[i]);
xp[row * DIMC + i] = half(v[j] * HAD_SCALE);
