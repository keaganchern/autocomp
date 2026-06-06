void test_rvv(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const struct xnn_qs8_rsum_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  // e8m2 and e32m8 have the same VLMAX.
  size_t vlmax = __riscv_vsetvlmax_e32m8();
  vint32m8_t vacc = __riscv_vmv_v_x_i32m8(0, vlmax);

  for (; batch > 0; ) {
    size_t vl = __riscv_vsetvl_e8m2(batch);
    vint8m2_t vt = __riscv_vle8_v_i8m2(input, vl);
    
    // Widen int8 directly to int32
    vint32m8_t vt32 = __riscv_vsext_vf4_i32m8(vt, vl);
    
    // Accumulate using tail-undisturbed policy.
    // This ensures that if vl < vlmax (in the final iteration), the parallel 
    // sums in the upper lanes of vacc from previous full iterations are preserved.
    vacc = __riscv_vadd_vv_i32m8_tu(vacc, vacc, vt32, vl);
    
    input += vl;
    batch -= vl;
  }

  // Reduce all parallel accumulators across the entire vlmax width
  vint32m1_t vres = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  vres = __riscv_vredsum_vs_i32m8_i32m1(vacc, vres, vlmax);
  
  *output += __riscv_vmv_x_s_i32m1_i32(vres);
}