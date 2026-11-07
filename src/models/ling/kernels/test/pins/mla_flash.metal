Stile.elems()[ii] *= float(scale2);
fg[loc] = ((col_pos + jj) < kL_rem) ? fg[loc] : neg_inf;
fg[loc] = (r < c) ? neg_inf : fg[loc];
Stile.template row_reduce<MaxOp>(new_max);
Stile.template row_bin_op<ExpSubOp>(new_max);
factor[i] = fast::exp2(max_score[i] - new_max[i]);
sum_score[i] = sum_score[i] * factor[i];
Stile.template row_reduce<SumOp>(sum_score);
Otile.template row_bin_op<MulOp>(factor);
rcp[i] = 1.f / sum_score[i];
Otile.template row_bin_op<MulOp>(rcp);
Otile.store_rows(O, int(O_strides[2]), lim_rows_q);
make_uniform(scale[0]) * make_uniform(1.44269504089f);
#define kernel kernel, max_total_threads_per_threadgroup(WM * WN * 32)
#undef kernel
