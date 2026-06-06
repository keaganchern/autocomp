// qu8-vadd-minmax: output[i] = clamp(round(scale_a*(a[i]-za) + scale_b*(b[i]-zb)) + zo,
//                                    min, max). Two u8 inputs, u8 output.
// Two-config sweep matching XNNPACK BinaryElementwiseParameters (utils.h:127-135).
#include "harness_common.h"

static const size_t configs[][1] = {
    {8192},
    {65536},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 65536

static uint8_t input_a_buf[MAX_BATCH];
static uint8_t input_b_buf[MAX_BATCH];
static uint8_t output_buf [MAX_BATCH];
static uint8_t gold_buf   [MAX_BATCH];

// Layout matches XNNPACK's struct xnn_qu8_add_minmax_params (microparams.h:333-345).
struct xnn_qu8_add_minmax_params {
    struct {
        uint8_t a_zero_point;
        uint8_t b_zero_point;
        int32_t bias;
        int32_t a_multiplier;
        int32_t b_multiplier;
        int32_t shift;
        int16_t output_zero_point;
        uint8_t output_min;
        uint8_t output_max;
    } scalar;
};
// Representative quantization params; values exercise both clamp bounds.
static const struct xnn_qu8_add_minmax_params qparams = {
    .scalar = {
        .a_zero_point = 128, .b_zero_point = 128,
        .bias = 0,
        .a_multiplier = 1 << 20, .b_multiplier = 1 << 20,
        .shift = 21,
        .output_zero_point = 128,
        .output_min = 16, .output_max = 240,
    }
};

typedef void (*candidate_fn_t)(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const struct xnn_qu8_add_minmax_params* params);

extern void reference_kernel(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const struct xnn_qu8_add_minmax_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_a_buf[i] = rand_u8();
    for (size_t i = 0; i < configs[si][0]; i++) input_b_buf[i] = rand_u8();
}

static void run_kernel(size_t si, candidate_fn_t fn, uint8_t* out) {
    fn(configs[si][0], input_a_buf, input_b_buf, out, &qparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, u8_bit_equal)
