#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/raddstoreexpminusmax.h"
#include "raddstoreexpminusmax-microkernel-tester.h"

static const size_t test_sizes[] = {3840, 32640};
#define NUM_SIZES (sizeof(test_sizes) / sizeof(test_sizes[0]))

typedef void (*candidate_fn_t)(
    size_t batch,
    const float* input,
    const float* max,
    float* output,
    float* sum,
    const void* params);

// SUBSTITUTE CANDIDATES
// SUBSTITUTE CANDIDATES END

int main() {
    volatile double dummy = 1.0;
    dummy = dummy / 1.0000001;

    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) {
        candidate_fn_t test_fn = candidate_fns[_ci];
        int _cand_id = candidate_ids[_ci];

        xnn_f32_raddstoreexpminusmax_ukernel_fn kernel_fn =
            (xnn_f32_raddstoreexpminusmax_ukernel_fn)test_fn;

        for (int _w = 0; _w < 5; _w++) {
            for (int _si = 0; _si < NUM_SIZES; _si++) {
                RAddStoreExpMinusMaxMicrokernelTester warmup;
                warmup.elements(test_sizes[_si]).iterations(1).Test(kernel_fn, nullptr);
            }
        }

        unsigned long total_cycles = 0;
        unsigned long total_instret = 0;
        for (int _si = 0; _si < NUM_SIZES; _si++) {
            size_t elems = test_sizes[_si];
            RAddStoreExpMinusMaxMicrokernelTester tester;
            tester.elements(elems)
                  .iterations(1)
                  .Test(kernel_fn, nullptr);
            if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
                printf("INCORRECT: candidate %d assertion failed (elems=%zu)\n", _cand_id, elems);
                if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD);
                goto next_candidate;
            }
            printf("  size %zu: %lu cycles, %lu instrs\n", elems, tester.kernel_cycles(), tester.kernel_instret());
            total_cycles += tester.kernel_cycles();
            total_instret += tester.kernel_instret();
        }

        printf("Correct result\n");
        printf("ID %d latency: %lu cycles\n", _cand_id, total_cycles);
        printf("ID %d instret: %lu instrs\n", _cand_id, total_instret);
        if (NUM_CANDIDATES == 1) {
            printf("Generated implementation latency: %lu cycles\n", total_cycles);
            printf("Generated implementation instret: %lu instrs\n", total_instret);
        }
        next_candidate:;
    }

    sys_reboot(SYS_REBOOT_COLD);
}
