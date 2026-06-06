void test_rvv(
    size_t batch,
    const float* input,
    uint16_t* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t n = batch / sizeof(float);
  uint16_t* o = output;

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);

    vfloat32m8_t vx = __riscv_vle32_v_f32m8(input, vl);

    vfloat32m8_t vabsx = __riscv_vfabs_v_f32m8(vx, vl);
    vuint32m8_t vabsx_u = __riscv_vreinterpret_v_f32m8_u32m8(vabsx);

    vuint32m8_t vbias = __riscv_vadd_vx_u32m8(vabsx_u, UINT32_C(0x07800000), vl);

    vfloat32m8_t vf = __riscv_vfmul_vf_f32m8(vabsx, 0x1.0p+112f, vl);
    vbool4_t vnanmask = __riscv_vmsgtu_vx_u32m8_b4(vabsx_u, UINT32_C(0x7F800000), vl);

    vbias = __riscv_vand_vx_u32m8(vbias, UINT32_C(0x7F800000), vl);
    vf = __riscv_vfmul_vf_f32m8(vf, 0x1.0p-110f, vl);

    vbias = __riscv_vmaxu_vx_u32m8(vbias, UINT32_C(0x40000000), vl);

    vf = __riscv_vfadd_vv_f32m8(vf, __riscv_vreinterpret_v_u32m8_f32m8(vbias), vl);

    vuint32m8_t vf_u = __riscv_vreinterpret_v_f32m8_u32m8(vf);
    vuint32m8_t vx_u = __riscv_vreinterpret_v_f32m8_u32m8(vx);

    vuint16m4_t vexph = __riscv_vnsrl_wx_u16m4(vf_u, 13, vl);
    vuint16m4_t vmanth = __riscv_vnsrl_wx_u16m4(vf_u, 0, vl);
    vuint16m4_t vsignh = __riscv_vnsrl_wx_u16m4(vx_u, 16, vl);

    vexph = __riscv_vand_vx_u16m4(vexph, UINT16_C(0x7C00), vl);
    vmanth = __riscv_vand_vx_u16m4(vmanth, UINT16_C(0x0FFF), vl);
    vsignh = __riscv_vand_vx_u16m4(vsignh, UINT16_C(0x8000), vl);

    vuint16m4_t vh = __riscv_vadd_vv_u16m4(vmanth, vexph, vl);

    vh = __riscv_vmerge_vxm_u16m4(vh, UINT16_C(0x7E00), vnanmask, vl);

    vh = __riscv_vor_vv_u16m4(vh, vsignh, vl);

    __riscv_vse16_v_u16m4(o, vh, vl);

    input += vl;
    o += vl;
    n -= vl;
  }
}