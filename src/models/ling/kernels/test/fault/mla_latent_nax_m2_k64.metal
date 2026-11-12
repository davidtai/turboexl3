// The m2_k64 fixture (mla_latent_test.zig, test only, never served): the NAX text's body at 64 keys a step with the
// whole scores (each simdgroup every one of the 36 slices), the shape nax_m2_k64 before strip2. The launch passes the
// served shape's arguments (MT 2: its grid and threads); these two definitions stand in for BKN and SD in the body.
#define BKN 64
#define SD 0
