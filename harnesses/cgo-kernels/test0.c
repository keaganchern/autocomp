// u8-vclamp: output[i] = clamp(input[i], min, max) over a batch of uint8 values.
// Bit-exact compare against the sol-injected `reference_kernel`.
//
// Two-config sweep matching XNNPACK's UnaryElementwiseParameters
// (XNNPACK/bench/utils.h:121-135): N ∈ {8192, 65536}. Headline metric is
// geomean(cycles/elem) × 1e6 — size-invariant.
#include "harness_common.h"

static const size_t configs[][1] = {
    {8192},
    {65536},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 65536  // largest config — sizes the static buffers

static uint8_t input_buf [MAX_BATCH];
static uint8_t output_buf[MAX_BATCH];
static uint8_t gold_buf  [MAX_BATCH];

// Layout matches XNNPACK's struct xnn_u8_minmax_params { scalar.{min,max}; }
// (XNNPACK/src/xnnpack/microparams.h:142). The sol only reads scalar.{min,max}.
struct xnn_u8_minmax_params { struct { uint32_t min, max; } scalar; };
static const struct xnn_u8_minmax_params minmax_params = {
    .scalar = { .min = 16, .max = 240 }   // both bounds active; ~6% top + ~6% bottom clipped
};

typedef void (*candidate_fn_t)(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_u8_minmax_params* params);

extern void reference_kernel(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const struct xnn_u8_minmax_params* params);

static size_t output_elems(size_t si) { return configs[si][0]; }

static void init_inputs(size_t si) {
    srand(0x1234);  // deterministic so gold reproduces across candidates
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_u8();
}

static void run_kernel(size_t si, candidate_fn_t fn, uint8_t* out) {
    fn(configs[si][0], input_buf, out, &minmax_params);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, u8_bit_equal)
