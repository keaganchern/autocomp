// f32-f16-vcvt: convert f32 input → f16 output (stored as uint16_t bit pattern).
// Two-config sweep matching XNNPACK UnaryElementwiseParameters. `batch` is in
// BYTES (kernel does n = batch / sizeof(float)); params is `const void*`.
//
// Inputs are bounded in [-1, 1] so the NaN-quieting branch in the sol stays
// dormant — keeps the bit-exact compare deterministic without exercising
// implementation-defined NaN payload behavior.
#include "harness_common.h"

static const size_t configs[][1] = {
    {2048},
    {16384},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 16384

static float    input_buf [MAX_BATCH];
static uint16_t output_buf[MAX_BATCH];
static uint16_t gold_buf  [MAX_BATCH];

typedef void (*candidate_fn_t)(
    size_t batch,
    const float* input,
    uint16_t* output,
    const void* params);

extern void reference_kernel(
    size_t batch,
    const float* input,
    uint16_t* output,
    const void* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_f1();
}

static void run_kernel(size_t si, candidate_fn_t fn, uint16_t* out) {
    fn(configs[si][0] * sizeof(float), input_buf, out, NULL);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, u16_bit_equal)
