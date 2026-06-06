// f32-vdiv: output[i] = input_a[i] / input_b[i]. Two-config sweep matching
// XNNPACK BinaryElementwiseParameters. `batch` is in BYTES.
// init guards input_b away from 0 so the bit-exact compare doesn't trip on
// platform-specific inf/NaN encodings.
#include "harness_common.h"

static const size_t configs[][1] = {
    {2048},
    {16384},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 16384

static float input_a_buf[MAX_BATCH];
static float input_b_buf[MAX_BATCH];
static float output_buf [MAX_BATCH];
static float gold_buf   [MAX_BATCH];

struct xnn_f32_default_params { char _unused; };
static const struct xnn_f32_default_params default_params = { 0 };

typedef void (*candidate_fn_t)(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* params);

extern void reference_kernel(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_a_buf[i] = rand_f1();
    // input_b held to [0.5, 1.5] in magnitude so quotient never overflows
    // and we never divide by zero.
    for (size_t i = 0; i < configs[si][0]; i++) {
        float v = rand_f1();
        input_b_buf[i] = (v >= 0 ? 0.5f + 0.5f * v : -0.5f + 0.5f * v);
    }
}

static void run_kernel(size_t si, candidate_fn_t fn, float* out) {
    fn(configs[si][0] * sizeof(float), input_a_buf, input_b_buf, out, &default_params);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_bit_equal)
