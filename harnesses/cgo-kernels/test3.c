// qu8-rsum: *output += sum(input[i]) over a batch of u8 values.
// Reduction kernel — single uint32_t output regardless of input size.
//
// Sizes from XNNPACK's ReduceParameters (utils.h:96-106) — the channels=1
// row is what a vector rsum bench typically sweeps. We pick the two
// extremes that exercise both warm-up overhead and steady-state.
#include "harness_common.h"

static const size_t configs[][1] = {
    {1024},   // small batch — prologue-dominated
    {8000},   // ReduceParameters' typical "long" row
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_BATCH 8000

static uint8_t  input_buf [MAX_BATCH];
static uint32_t output_buf[1];
static uint32_t gold_buf  [1];

// Layout matches XNNPACK's struct xnn_qs8_rsum_params (microparams.h:419-421).
// Used unchanged by both qs8 and qu8 rsum kernels.
struct xnn_qs8_rsum_params { char _; };
static const struct xnn_qs8_rsum_params rparams = { 0 };

typedef void (*candidate_fn_t)(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* params);

extern void reference_kernel(
    size_t batch,
    const uint8_t* input,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* params);

// Comparison element count: 1 (single u32 reduction output). The macro
// also uses this as the cycles-per-elem denominator — for rsum the
// reported "cycles/elem" is really cycles-per-reduction (not per-input
// byte), which means score numbers are not directly comparable to other
// kernels but still produce a clean ranking within rsum's search.
static size_t output_elems(size_t si) { (void)si; return 1; }

static void init_inputs(size_t si) {
    srand(0x1234);
    for (size_t i = 0; i < configs[si][0]; i++) input_buf[i] = rand_u8();
}

static void run_kernel(size_t si, candidate_fn_t fn, uint32_t* out) {
    // rsum is a *= +=  accumulator (*output += sum(input)); the macro's
    // 3-iter warmup loop would otherwise leave GOLD = 3*sum vs the
    // candidate's per-iter-zeroed 1*sum. Zero here for both paths.
    *out = 0;
    fn(configs[si][0], input_buf, out, &rparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, u32_bit_equal)
