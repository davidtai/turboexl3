// turboexl3_ling_mla_attn_nax's header after MLX steel's NAX tiles (core/steel.zig): the row ops its online softmax
// folds the score tiles with (the max, the sum, the rescale, and exp(x - max) in natural units, the units launch 3's
// merge reads pm in), the finite minimum a masked score takes (MLX's attention_nax: -numeric_limits<float>::max(), so a
// rescale of a row that has seen no key is exp(-max) = 0, never NaN), then the attribute MLX's kernel declares, its
// threads per threadgroup (MT M tiles of 4 simdgroups), put on the generated [[kernel]] definition (the body's last
// piece undefines it, so MLX's explicit instantiation keeps a plain [[kernel]]).
using namespace mlx::steel;

struct NaxMaxOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::max(x, y);
  }
};

struct NaxSumOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x + y;
  }
};

struct NaxMulOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return x * y;
  }
};

struct NaxExpSubOp {
  template <typename T>
  METAL_FUNC static constexpr T apply(T x, T y) {
    return metal::exp(x - y);
  }
};

constant constexpr float mla_nax_finite_min = -metal::numeric_limits<float>::max();

#define kernel kernel, max_total_threads_per_threadgroup(MT * 128)
