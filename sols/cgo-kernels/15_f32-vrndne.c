void test_rvv(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* params) 
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl);

    vfloat32m8_t vabsx = __riscv_vfabs_v_f32m8(vx, vl);
    
    // vcaltq_f32(vmagic_number, vx) -> abs(magic) < abs(vx)
    vbool4_t cmp = __riscv_vmfgt_vf_f32m8_b4(vabsx, 8388608.0f, vl);

    vfloat32m8_t vrndabsx = __riscv_vfadd_vf_f32m8(vabsx, 8388608.0f, vl);
    vrndabsx = __riscv_vfsub_vf_f32m8(vrndabsx, 8388608.0f, vl);

    // Combine sign bit of vx with bits 0-30 of vrndabsx
    vfloat32m8_t vrnd_signed = __riscv_vfsgnj_vv_f32m8(vrndabsx, vx, vl);
    
    // Select vx if cmp is true, else vrnd_signed
    vfloat32m8_t vy = __riscv_vmerge_vvm_f32m8(vrnd_signed, vx, cmp, vl);

    // NaN fixup: NEON propagates the NaN payload and quiets it.
    // RVV arithmetic canonicalizes NaNs, so we manually restore the quieted original NaN.
    vbool4_t is_nan = __riscv_vmfne_vv_f32m8_b4(vx, vx, vl);
    vuint32m8_t vx_u32 = __riscv_vreinterpret_v_f32m8_u32m8(vx);
    vuint32m8_t vx_quiet = __riscv_vor_vx_u32m8(vx_u32, UINT32_C(0x00400000), vl);
    vfloat32m8_t vx_qnan = __riscv_vreinterpret_v_u32m8_f32m8(vx_quiet);
    vy = __riscv_vmerge_vvm_f32m8(vy, vx_qnan, is_nan, vl);

    __riscv_vse32_v_f32m8(output, vy, vl);

    input += vl;
    output += vl;
    n -= vl;
  }
}