// qs8-f32-vcvt: dequantize int8 input → f32 output
// (output = (input - zero_point) * scale).
// Two-config sweep matching XNNPACK UnaryElementwiseParameters (utils.h:121-125).
// `batch` is in BYTES (sizeof(int8_t) == 1, so it equals element count here).
#include "harness_common.h"

static const size_t configs[][1] = {
    {8192},
    {65536},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 65536

static int8_t input_buf [MAX_BATCH];
static float  output_buf[MAX_BATCH];
static float  gold_buf  [MAX_BATCH];

// Layout matches XNNPACK's struct xnn_qs8_f32_cvt_params (microparams.h:501-506).
struct xnn_qs8_f32_cvt_params {
    struct {
        int32_t zero_point;
        float   scale;
    } scalar;
};
// Representative dequantization params: zp=0, scale=1/128 maps s8 to ~[-1, 1].
static const struct xnn_qs8_f32_cvt_params qparams = {
    .scalar = {
        .zero_point = 0,
        .scale      = 1.0f / 128.0f,
    }
};

typedef void (*candidate_fn_t)(
    size_t batch,
    const int8_t* input,
    float* output,
    const struct xnn_qs8_f32_cvt_params* params);

extern void reference_kernel(
    size_t batch,
    const int8_t* input,
    float* output,
    const struct xnn_qs8_f32_cvt_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_s8();
}

static void run_kernel(size_t si, candidate_fn_t fn, float* out) {
    fn(configs[si][0], input_buf, out, &qparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_bit_equal)
