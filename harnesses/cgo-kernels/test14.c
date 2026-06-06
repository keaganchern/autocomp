// f32-vmulc: scalar-broadcast multiply. output[i] = input_a[i] * (*input_b),
// where input_b is a 1-element vector holding the broadcast scalar. `batch` is
// in BYTES of input_a. Two-config sweep.
#include "harness_common.h"

static const size_t configs[][1] = {
    {2048},
    {16384},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 16384

static float input_a_buf[MAX_BATCH];
static float input_b_scalar = 0.0f;   // broadcast multiplier; init'd per config
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
    input_b_scalar = rand_f1();
}

static void run_kernel(size_t si, candidate_fn_t fn, float* out) {
    fn(configs[si][0] * sizeof(float), input_a_buf, &input_b_scalar, out, &default_params);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_bit_equal)
