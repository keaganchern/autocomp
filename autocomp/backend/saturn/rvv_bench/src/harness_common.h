// Shared boilerplate for saturn-rvv test harnesses.
//
// Each harness lives in harnesses/<prob_type>/test<N>.c and gets copied
// into this build slot as src/main.c by saturn_eval._build_and_run_spike.
// The harness #includes this header, declares its kernel signature and
// buffers, writes init_inputs() + run_kernel(), and ends with one
// HARNESS_MAIN(...) line.
//
// All parser contracts required by saturn_eval.py:565-571 live here:
//   - "Correct result"
//   - "ID %d latency: %lu cycles"
//   - "Generated implementation latency: %lu cycles"  (when NUM_CANDIDATES==1)
//   - sys_reboot(SYS_REBOOT_COLD) on success and on single-candidate failure

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

// ---------- helpers ----------

static unsigned long read_cycles(void) {
    unsigned long c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c;
}
static unsigned long read_instret(void) {
    unsigned long v; __asm__ volatile("rdinstret %0" : "=r"(v)); return v;
}
static inline void fence(void) {
    __asm__ volatile("fence" ::: "memory");
}
static inline void vsetvli_reset(void) {
    __asm__ volatile("vsetvli x0, x0, e8, m1, ta, ma");
}
static inline float rand_f1(void) {
    // [-1, 1]; deterministic given srand seed.
    return -1.0f + 2.0f * ((float)rand() / (float)RAND_MAX);
}
static inline uint8_t  rand_u8 (void) { return (uint8_t) (rand() & 0xff); }
static inline int8_t   rand_s8 (void) { return (int8_t)  (rand() & 0xff); }

// ---------- comparisons ----------

// Bit-exact: every output word must match. Use when the candidate is expected
// to preserve the reference's FP order (vmul, transpose, per-row GEMV, etc).
static int f32_bit_equal(const float *got, const float *ref, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t a, b;
        __builtin_memcpy(&a, &got[i], 4);
        __builtin_memcpy(&b, &ref[i], 4);
        if (a != b) {
            printf("Mismatch @i=%zu got=0x%08x (%f) expected=0x%08x (%f)\n",
                   i, a, (double)got[i], b, (double)ref[i]);
            return 0;
        }
    }
    return 1;
}

// Bit-exact byte compare. Use for any integer-typed kernel (u8, s8, i32 …).
static int u8_bit_equal(const uint8_t *got, const uint8_t *ref, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (got[i] != ref[i]) {
            printf("Mismatch @i=%zu got=0x%02x (%u) expected=0x%02x (%u)\n",
                   i, got[i], got[i], ref[i], ref[i]);
            return 0;
        }
    }
    return 1;
}

static int s8_bit_equal(const int8_t *got, const int8_t *ref, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (got[i] != ref[i]) {
            printf("Mismatch @i=%zu got=0x%02x (%d) expected=0x%02x (%d)\n",
                   i, (uint8_t)got[i], got[i], (uint8_t)ref[i], ref[i]);
            return 0;
        }
    }
    return 1;
}

// Tolerant: |got - ref| <= abs_tol + rel_tol*|ref|.
// Use when reductions may reorder (GEMM tiled differently across LMUL etc).
static int f32_approx_equal_tol(const float *got, const float *ref, size_t n,
                                float abs_tol, float rel_tol) {
    for (size_t i = 0; i < n; i++) {
        float d = got[i] - ref[i]; if (d < 0) d = -d;
        float ra = ref[i] < 0 ? -ref[i] : ref[i];
        float tol = abs_tol + rel_tol * ra;
        if (d > tol) {
            uint32_t a, b;
            __builtin_memcpy(&a, &got[i], 4);
            __builtin_memcpy(&b, &ref[i], 4);
            printf("Mismatch @i=%zu got=0x%08x (%f) expected=0x%08x (%f) diff=%g tol=%g\n",
                   i, a, (double)got[i], b, (double)ref[i], (double)d, (double)tol);
            return 0;
        }
    }
    return 1;
}

// Default tolerance for GEMM-class kernels (K reductions over U[-1,1]).
// Sized for K ~ 512 — bump abs_tol if you use much larger K.
static int f32_approx_equal_default(const float *got, const float *ref, size_t n) {
    return f32_approx_equal_tol(got, ref, n, 1e-4f, 1e-5f);
}

// ---------- measurement window ----------

// Cycle/instret window around an arbitrary statement block.
// Fence + vsetvli reset on both sides to flush ROB and clear vector state.
#define MEASURE(cycles_out, instret_out, BODY) do { \
    fence(); vsetvli_reset(); \
    unsigned long _meas_sc = read_cycles(); \
    unsigned long _meas_si = read_instret(); \
    BODY; \
    fence(); vsetvli_reset(); \
    unsigned long _meas_ei = read_instret(); \
    unsigned long _meas_ec = read_cycles(); \
    (cycles_out) = _meas_ec - _meas_sc; \
    (instret_out) = _meas_ei - _meas_si; \
} while (0)

// ---------- parser-required emitters ----------

// Single point of truth for the lines saturn_eval.py scans.
#define EMIT_PASS(cand_id, cycles, instret) do { \
    printf("Correct result\n"); \
    printf("ID %d latency: %lu cycles\n",  (int)(cand_id), (unsigned long)(cycles)); \
    printf("ID %d instret: %lu instrs\n",  (int)(cand_id), (unsigned long)(instret)); \
    if (NUM_CANDIDATES == 1) { \
        printf("Generated implementation latency: %lu cycles\n",  (unsigned long)(cycles)); \
        printf("Generated implementation instret: %lu instrs\n",  (unsigned long)(instret)); \
    } \
} while (0)

#define FAIL_AND_NEXT(cand_id, FMT, ...) do { \
    printf("INCORRECT: candidate %d failed " FMT "\n", (int)(cand_id), ##__VA_ARGS__); \
    if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD); \
    goto next_candidate; \
} while (0)

// NUM_CANDIDATES, candidate_fn_t, candidate_fns[], candidate_ids[] are
// supplied by saturn's SUBSTITUTE HERE block — the harness must place
// those markers somewhere between this header and HARNESS_MAIN(), and the
// candidate_fn_t typedef must be declared by the harness above the markers.

// ---------- HARNESS_MAIN ----------
//
// Expands to a full main() that runs every injected candidate through:
//   warmup → gold pass → timed pass → compare → emit.
// The harness must provide:
//   static void init_inputs(void);
//   static void run_kernel(candidate_fn_t fn, float *out);
//
// Parameters:
//   REF      — reference function symbol, called via run_kernel(REF, GOLD).
//   OUT      — output buffer pointer (per-call destination).
//   GOLD     — gold buffer pointer (reference output destination).
//   N_OUT    — element count for compare + memset.
//   COMPARE  — predicate (got, ref, n) -> int; e.g. f32_bit_equal.
//   FAIL_FMT — printf format string with extra context on failure
//              (e.g. "(m=%d k=%d n=%d)"); variadic args follow.
//
// Example:
//   HARNESS_MAIN(reference_gemm, c_out, c_gold, M*NC, f32_bit_equal,
//                "(m=%d kc=%d nc=%d)", M, KC, NC)
//
#define HARNESS_MAIN(REF, OUT, GOLD, N_OUT, COMPARE, FAIL_FMT, ...) \
int main(void) { \
    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) { \
        candidate_fn_t _fn = candidate_fns[_ci]; \
        int _id = candidate_ids[_ci]; \
        /* Warm reference: setup once, then 3 back-to-back runs so the */ \
        /* scalar code (rand, memset) doesn't thrash I-cache between calls. */ \
        init_inputs(); \
        memset((GOLD), 0, (size_t)(N_OUT) * sizeof(*(OUT))); \
        for (int _w = 0; _w < 3; _w++) { run_kernel((REF), (GOLD)); } \
        /* Time candidate 3 times, take min — selects the warmest-cache */ \
        /* iteration as the steady-state representative. The first run */ \
        /* acts as warmup; subsequent runs reflect amortized cold-start. */ \
        unsigned long _cyc = (unsigned long)-1, _ins = 0; \
        for (int _w = 0; _w < 3; _w++) { \
            init_inputs(); \
            memset((OUT), 0, (size_t)(N_OUT) * sizeof(*(OUT))); \
            unsigned long _cyc_i, _ins_i; \
            MEASURE(_cyc_i, _ins_i, { run_kernel(_fn, (OUT)); }); \
            if (_cyc_i < _cyc) { _cyc = _cyc_i; _ins = _ins_i; } \
        } \
        if (!(COMPARE)((OUT), (GOLD), (size_t)(N_OUT))) \
            FAIL_AND_NEXT(_id, FAIL_FMT, ##__VA_ARGS__); \
        EMIT_PASS(_id, _cyc, _ins); \
        next_candidate:; \
    } \
    sys_reboot(SYS_REBOOT_COLD); \
    return 0; \
}

// ---------- HARNESS_MAIN_MULTI ----------
//
// Multi-config sweep. Drives the candidate through NUM_CFGS operating points
// and reports a size-invariant headline score = geomean(cycles/elem) × 1e6
// (lower = better, integer). saturn_eval.py prefers this over "latency" when
// both are emitted, so the search optimizer naturally uses geomean.
//
// Required helpers (harness must define):
//   static size_t output_elems(size_t si);                       // elems for config si
//   static void   init_inputs (size_t si);                       // fill input_buf for config si
//   static void   run_kernel  (size_t si, candidate_fn_t fn, T* out);  // call fn for config si
//
// Parameters mirror HARNESS_MAIN, minus FAIL_FMT (the macro emits a fixed
// "config %zu (n=%zu)" failure line; per-config context lives in run_kernel
// debug output if you need it).
//
// Example:
//   HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf,
//                      NUM_CONFIGS, u8_bit_equal)
//
// Integer nth-root using exponentiation by squaring + binary search.
// Returns floor(value ** (1/n)). Used by HARNESS_MAIN_MULTI for geomean;
// pure integer math so we avoid libm's autovec-on-spike-incompatible code paths.
static unsigned long _harness_ipow_lim(unsigned long base, unsigned int n, unsigned long lim) {
    unsigned long r = 1;
    for (unsigned int i = 0; i < n; i++) {
        if (base != 0 && r > lim / base) return lim + 1;
        r *= base;
    }
    return r;
}
static unsigned long _harness_iroot(unsigned long value, unsigned int n) {
    if (n == 1 || value <= 1) return value;
    unsigned long lo = 1, hi = value;
    while (lo < hi) {
        unsigned long mid = lo + (hi - lo + 1) / 2;
        if (_harness_ipow_lim(mid, n, value) <= value) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// Multi-config sweep. Drives the candidate through NUM_CFGS operating points
// and reports a size-invariant headline score = geomean(cycles/elem) × 1e6
// (lower = better, integer). saturn_eval.py prefers this over "latency" when
// both are emitted, so the search optimizer naturally uses geomean.
//
// Required helpers (harness must define):
//   static size_t output_elems(size_t si);                       // elems for config si
//   static void   init_inputs (size_t si);                       // fill input_buf for config si
//   static void   run_kernel  (size_t si, candidate_fn_t fn, T* out);  // call fn for config si
//
// Parameters mirror HARNESS_MAIN, minus FAIL_FMT (the macro emits a fixed
// "config %zu (n=%zu)" failure line; per-config context lives in run_kernel
// debug output if you need it).
//
// Example:
//   HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf,
//                      NUM_CONFIGS, u8_bit_equal)
//
// NOTE: NUM_CFGS must be ≤ 16; cpe values up to ~1e7 fit in u64 product space.
#define HARNESS_MAIN_MULTI(REF, OUT, GOLD, NUM_CFGS, COMPARE) \
int main(void) { \
    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) { \
        candidate_fn_t _fn = candidate_fns[_ci]; \
        int _id = candidate_ids[_ci]; \
        unsigned long _total_cyc = 0, _total_ins = 0; \
        unsigned long _cpe_prod = 1; \
        int _failed = 0; \
        for (size_t _si = 0; _si < (size_t)(NUM_CFGS); _si++) { \
            size_t _n = output_elems(_si); \
            /* Warm reference: setup once, then 3 back-to-back runs so the */ \
            /* scalar code (rand, memset) doesn't thrash I-cache between calls. */ \
            init_inputs(_si); \
            memset((GOLD), 0, _n * sizeof(*(OUT))); \
            for (int _w = 0; _w < 3; _w++) { run_kernel(_si, (REF), (GOLD)); } \
            /* Time candidate 3 times, take min — selects the warmest-cache */ \
            /* iteration. First run acts as warmup; min reflects steady-state. */ \
            unsigned long _cyc = (unsigned long)-1, _ins = 0; \
            for (int _w = 0; _w < 3; _w++) { \
                init_inputs(_si); \
                memset((OUT), 0, _n * sizeof(*(OUT))); \
                unsigned long _cyc_i, _ins_i; \
                MEASURE(_cyc_i, _ins_i, { run_kernel(_si, _fn, (OUT)); }); \
                if (_cyc_i < _cyc) { _cyc = _cyc_i; _ins = _ins_i; } \
            } \
            if (!(COMPARE)((OUT), (GOLD), _n)) { \
                printf("INCORRECT: candidate %d failed config %zu (n=%zu)\n", \
                       _id, _si, _n); \
                _failed = 1; break; \
            } \
            unsigned long _cpe = ((unsigned long)_cyc * 1000000UL) / (_n ? _n : 1); \
            _total_cyc += _cyc; _total_ins += _ins; \
            _cpe_prod *= (_cpe ? _cpe : 1); \
            printf("  config %zu n=%zu: %lu cycles, %lu instrs, %lu.%06lu cycles/elem\n", \
                   _si, _n, _cyc, _ins, _cpe / 1000000UL, _cpe % 1000000UL); \
        } \
        if (_failed) { \
            if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD); \
            goto next_candidate; \
        } \
        unsigned long _score = _harness_iroot(_cpe_prod, (unsigned int)(NUM_CFGS)); \
        printf("Correct result\n"); \
        printf("ID %d latency: %lu cycles\n", _id, _total_cyc); \
        printf("ID %d instret: %lu instrs\n", _id, _total_ins); \
        printf("ID %d score: %lu (geomean cycles/elem x1e6)\n", _id, _score); \
        if (NUM_CANDIDATES == 1) { \
            printf("Generated implementation latency: %lu cycles\n", _total_cyc); \
            printf("Generated implementation instret: %lu instrs\n", _total_ins); \
            printf("Generated implementation score: %lu\n", _score); \
        } \
        next_candidate:; \
    } \
    sys_reboot(SYS_REBOOT_COLD); \
    return 0; \
}
