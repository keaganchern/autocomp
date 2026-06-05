#include <cstdio>
#include <cstdlib>

#include <riscv_vector.h>
#include <zephyr/sys/reboot.h>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "transposec-microkernel-tester.h"

static const size_t test_sizes[] = {128, 256, 512};
#define NUM_SIZES (sizeof(test_sizes) / sizeof(test_sizes[0]))

typedef void (*candidate_fn_t)(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

// SUBSTITUTE CANDIDATES
// SUBSTITUTE CANDIDATES END

int main() {
    for (int _ci = 0; _ci < NUM_CANDIDATES; _ci++) {
        candidate_fn_t test_fn = candidate_fns[_ci];
        int _cand_id = candidate_ids[_ci];

        // Warmup: prime TLB, branch predictor, and L2 cache
        for (int _w = 0; _w < 5; _w++) {
            for (int _si = 0; _si < NUM_SIZES; _si++) {
                TransposecMicrokernelTester warmup;
                warmup.iterations(1)
                      .block_height(test_sizes[_si])
                      .block_width(test_sizes[_si])
                      .input_stride(test_sizes[_si])
                      .output_stride(test_sizes[_si])
                      .Test(test_fn);
            }
        }

        unsigned long total_cycles = 0;
        unsigned long total_instret = 0;
        // Normalized score: sum of (cycles * 1000 / elements) across sizes.
        // Each size contributes roughly equally regardless of absolute cycle count.
        unsigned long score = 0;

        for (int _si = 0; _si < NUM_SIZES; _si++) {
            size_t sz = test_sizes[_si];
            size_t num_elements = sz * sz;
            TransposecMicrokernelTester tester;
            tester.iterations(1)
                  .block_height(sz)
                  .block_width(sz)
                  .input_stride(sz)
                  .output_stride(sz)
                  .Test(test_fn);

            if (testing::UnitTest::GetInstance()->ad_hoc_test_result().Failed()) {
                printf("INCORRECT: candidate %d assertion failed (size=%zu)\n", _cand_id, sz);
                if (NUM_CANDIDATES == 1) sys_reboot(SYS_REBOOT_COLD);
                goto next_candidate;
            }
            unsigned long cycles = tester.kernel_cycles();
            unsigned long instret = tester.kernel_instret();
            printf("  size %zu: %lu cycles, %lu instrs, %lu.%03lu cycles/elem\n",
                   sz, cycles, instret,
                   cycles / num_elements, (cycles * 1000 / num_elements) % 1000);
            total_cycles += cycles;
            total_instret += instret;
            score += (cycles * 1000UL) / num_elements;
        }

        printf("Correct result\n");
        printf("ID %d latency: %lu cycles\n", _cand_id, total_cycles);
        printf("ID %d instret: %lu instrs\n", _cand_id, total_instret);
        printf("ID %d score: %lu\n", _cand_id, score);
        if (NUM_CANDIDATES == 1) {
            printf("Generated implementation latency: %lu cycles\n", total_cycles);
            printf("Generated implementation instret: %lu instrs\n", total_instret);
            printf("Generated implementation score: %lu\n", score);
        }
        next_candidate:;
    }

    sys_reboot(SYS_REBOOT_COLD);
}
