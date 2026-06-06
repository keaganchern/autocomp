// qs8-vcvt: requantize int8 input → int8 output via affine transform
// (output = clamp((input - input_zp) * multiplier >> shift + output_zp, -128, 127)).
// Two-config sweep matching XNNPACK UnaryElementwiseParameters (utils.h:121-125).
#include "harness_common.h"

static const size_t configs[][1] = {
    {8192},
    {65536},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 65536

static int8_t input_buf [MAX_BATCH];
static int8_t output_buf[MAX_BATCH];
static int8_t gold_buf  [MAX_BATCH];

// Layout matches XNNPACK's struct xnn_qs8_cvt_params (microparams.h:486-492).
struct xnn_qs8_cvt_params {
    struct {
        int16_t input_zero_point;
        int32_t multiplier;
        int16_t output_zero_point;
    } scalar;
};
// Representative requantization params (scale ≈ 1.0 in fixed-point).
static const struct xnn_qs8_cvt_params qparams = {
    .scalar = {
        .input_zero_point  = 0,
        .multiplier        = 0x40000000,   // ~1.0 in Q1.30
        .output_zero_point = 0,
    }
};

typedef void (*candidate_fn_t)(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* params);

extern void reference_kernel(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_s8();
}

static void run_kernel(size_t si, candidate_fn_t fn, int8_t* out) {
    fn(configs[si][0], input_buf, out, &qparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, s8_bit_equal)
