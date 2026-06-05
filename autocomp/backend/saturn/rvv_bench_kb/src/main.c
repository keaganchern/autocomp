/*
 * Placeholder main.c for rvv_bench_kb.
 *
 * This file is OVERWRITTEN on every build by saturn_eval.py with the harness +
 * autocomp candidate code (see _build_and_run_spike at saturn_eval.py:164).
 * The contents below only matter if the app is built directly without autocomp.
 *
 * Per-problem data files (data.h, model_pte.c, input_data.c) are populated by
 * scripts/extract_kb_problem.py before any autocomp run. Without that step,
 * the linker will pull in the empty placeholders shipped with this directory
 * and the binary will be non-functional — only useful as a CMake liveness check.
 */

#include <stdio.h>
#include <zephyr/sys/reboot.h>

int main(void) {
    printf("rvv_bench_kb placeholder — run scripts/extract_kb_problem.py first\n");
    sys_reboot(SYS_REBOOT_COLD);
    return 0;
}
