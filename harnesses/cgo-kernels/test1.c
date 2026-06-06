// s8-vclamp: output[i] = clamp(input[i], min, max) over a batch of int8 values.
// Two-config sweep matching XNNPACK's UnaryElementwiseParameters
// (XNNPACK/bench/utils.h:121-135): N ∈ {8192, 65536}.
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

// Layout matches XNNPACK's struct xnn_s8_minmax_params (microparams.h:135-139).
struct xnn_s8_minmax_params { struct { int32_t min, max; } scalar; };
static const struct xnn_s8_minmax_params minmax_params = {
    .scalar = { .min = -100, .max = 100 }   // ~22% top + 22% bottom clipped
};

typedef void (*candidate_fn_t)(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_s8_minmax_params* params);

extern void reference_kernel(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_s8_minmax_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_s8();
}

static void run_kernel(size_t si, candidate_fn_t fn, int8_t* out) {
    fn(configs[si][0], input_buf, out, &minmax_params);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, s8_bit_equal)
