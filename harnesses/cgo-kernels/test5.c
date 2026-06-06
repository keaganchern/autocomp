// qs8-vlrelu: leaky-ReLU on int8 — input split into positive/negative branches,
// each scaled by its own multiplier, requantized to int8 output.
// Two-config sweep matching XNNPACK UnaryElementwiseParameters.
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

// Layout matches XNNPACK's struct xnn_qs8_lrelu_params (microparams.h:555-562).
struct xnn_qs8_lrelu_params {
    struct {
        int32_t input_zero_point;
        int32_t positive_multiplier;
        int32_t negative_multiplier;
        int32_t output_zero_point;
    } scalar;
};
// Negative slope ≈ 0.1 (multiplier ratio). Positive slope ≈ 1.0.
static const struct xnn_qs8_lrelu_params lrparams = {
    .scalar = {
        .input_zero_point    = 0,
        .positive_multiplier = 0x40000000,
        .negative_multiplier = 0x06666666,   // ~0.1 in Q1.30
        .output_zero_point   = 0,
    }
};

typedef void (*candidate_fn_t)(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_lrelu_params* params);

extern void reference_kernel(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_lrelu_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_s8();
}

static void run_kernel(size_t si, candidate_fn_t fn, int8_t* out) {
    fn(configs[si][0], input_buf, out, &lrparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, s8_bit_equal)
