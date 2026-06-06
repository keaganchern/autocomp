// f32-vlrelu: output[i] = (input[i] < 0) ? slope*input[i] : input[i].
// Two-config sweep matching XNNPACK UnaryElementwiseParameters. `batch` is in BYTES.
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

// Layout matches XNNPACK's struct xnn_f32_lrelu_params (microparams.h).
struct xnn_f32_lrelu_params { struct { float slope; } scalar; };
static const struct xnn_f32_lrelu_params lparams = {
    .scalar = { .slope = 0.1f }
};

typedef void (*candidate_fn_t)(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_lrelu_params* params);

extern void reference_kernel(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_lrelu_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_f1();
}

static void run_kernel(size_t si, candidate_fn_t fn, float* out) {
    fn(configs[si][0] * sizeof(float), input_buf, out, &lparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_bit_equal)
