// f32-vrndne: round-to-nearest-even on each f32 input. Two-config sweep matching
// XNNPACK UnaryElementwiseParameters. `batch` is in BYTES.
//
// rand_f1() returns [-1, 1] — well below the 2^23 round-via-magic-add threshold —
// so the sol's NaN-quieting branch is never exercised in this harness. That keeps
// the bit-exact compare deterministic across LMUL choices.
#include "harness_common.h"

static const size_t configs[][1] = {
    {2048},
    {16384},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 16384

static float input_buf [MAX_BATCH];
static float output_buf[MAX_BATCH];
static float gold_buf  [MAX_BATCH];

struct xnn_f32_default_params { char _unused; };
static const struct xnn_f32_default_params default_params = { 0 };

typedef void (*candidate_fn_t)(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* params);

extern void reference_kernel(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    // Scale by 4 so a non-trivial fraction land near .5 boundaries — exercises
    // the round-to-even semantics rather than just the identity path on integers.
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = 4.0f * rand_f1();
}

static void run_kernel(size_t si, candidate_fn_t fn, float* out) {
    fn(configs[si][0] * sizeof(float), input_buf, out, &default_params);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_bit_equal)
