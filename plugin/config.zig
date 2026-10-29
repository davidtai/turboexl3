//! The kit's `test` step builds this file over the kit alone. The quant table's claim, validation and bill, and their
//! MLX-free tests, are turboexl3_table's (`table/config.zig`); the NAX gate's tests are turboexl3_kit's (`kit/nax.zig`).
//! A model plugin that builds the table in runs them in its own MLX-free test step.
