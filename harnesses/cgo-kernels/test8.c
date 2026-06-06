// f32-vcmul: complex multiply. Layout — input_a / input_b / output each store
// the real half at offset 0 and the imaginary half at offset `batch` bytes.
// So each buffer holds 2*n floats total, and `batch` is the per-channel byte count.
//
// Two-config sweep — per-channel n; total floats per buffer = 2*n. Score is
// reported as cycles per total float written (2*n elements).
#include "harness_common.h"

static const size_t configs[][1] = {
    {1024},   // per-channel n; 2048 total f32 written
    {8192},   // per-channel n; 16384 total f32 written
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 8192
#define BUF_SIZE  (2 * MAX_BATCH)   // real + imag halves laid out back-to-back

static float input_a_buf[BUF_SIZE];
static float input_b_buf[BUF_SIZE];
static float output_buf [BUF_SIZE];
static float gold_buf   [BUF_SIZE];

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

// Score on total f32 written (real + imag). Keeps cycles/elem comparable to
// other elementwise f32 kernels in this suite.
static size_t output_elems(size_t si) { return 2 * configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t total = 2 * configs[si][0];
    for (size_t i = 0; i < total; i++) input_a_buf[i] = rand_f1();
    for (size_t i = 0; i < total; i++) input_b_buf[i] = rand_f1();
}

static void run_kernel(size_t si, candidate_fn_t fn, float* out) {
    fn(configs[si][0] * sizeof(float), input_a_buf, input_b_buf, out, &default_params);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_bit_equal)
