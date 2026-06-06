void test_rvv(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_lrelu_params* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  const float slope = params->scalar.slope;

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl);

    vfloat32m8_t vacc = __riscv_vfmul_vf_f32m8(vx, slope, vl);
    
    vint32m8_t vx_i = __riscv_vreinterpret_v_f32m8_i32m8(vx);
    vbool4_t mask = __riscv_vmslt_vx_i32m8_b4(vx_i, 0, vl);

    vfloat32m8_t vout = __riscv_vmerge_vvm_f32m8(vx, vacc, mask, vl);

    __riscv_vse32_v_f32m8(output, vout, vl);

    input += vl;
    output += vl;
    n -= vl;
  }
}