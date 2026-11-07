// turboexl3_ling_mla_flash's header after MLX steel's NAX tiles (core/steel.zig): from MLX 0.32.5's
// steel/attn/kernels/steel_attention_nax.h (Copyright © 2024-26 Apple Inc., MIT), the row ops attention_nax folds its
// tiles with (MaxOp, SumOp, MulOp, ExpSubOp, verbatim) and its Limits<float>::finite_min (MLX's utils.h:
// -numeric_limits<float>::max()); then the attribute MLX's kernel declares, its threads per threadgroup, put on the
// generated [[kernel]] definition (the body's last piece undefines it, so MLX's explicit instantiation keeps a plain
// [[kernel]]).
using namespace mlx::steel;

struct MaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct SumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct MulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct ExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return fast::exp2(x - y);
  }
};

constant constexpr float mla_flash_finite_min = -metal::numeric_limits<float>::max();

#define kernel kernel, max_total_threads_per_threadgroup(WM * WN * 32)
