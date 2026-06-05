---
name: test-harness
description: Compile a saturn-rvv test harness with its reference sol injected as the candidate, then run on spike (skips firesim). Use when the user asks to compile-check, smoke-test, validate, or "see if it compiles" a harness file under harnesses/<prob_type>/test<N>.c — typically after writing or editing one and before kicking off a full search. Reports compile errors, spike hangs/timeouts, and bit-exact / tolerant mismatch failures, plus the cycle count on success.
---

# test-harness

Runs the saturn_eval.py spike path on a single harness with the corresponding sol injected as candidate 0. No firesim, no LLM mutation, no search loop — just: write `main.c`, `west build -b spike_riscv64`, `spike`.

## Invocation

```bash
bash -lc 'source /scratch/kchern2/chipyard/env.sh && python3 .claude/skills/test-harness/run.py <harness_path> [sol_path]'
```

- `<harness_path>`: absolute or repo-relative path like `harnesses/xnnpack-f32/test9.c`. Filename must be `test<N>.c`.
- `[sol_path]`: optional override; defaults to `sols/<prob_type>/<N>_*.c`.

The chipyard env source is required so `spike` is on PATH. The zephyr SDK env is sourced inside the build subprocess by saturn_eval itself.

## What it does (mirrors saturn_eval `_build_and_run_spike`)

1. Parses `prob_type` and `prob_id` from the harness path.
2. Locates the sol, cleans it via `clean_code`, extracts param sig via `SaturnEvalBackend._extract_candidate_params`.
3. Calls `SaturnTest(harness).inject_candidates([sol_body], [0], param_sig)` to materialize a full `main.c`.
4. Calls `_build_and_run_spike((code, slot=0, idx=0, prob_type))` — same code path as the search.
5. Prints spike stdout; exits 0 on `Correct result`, 1 on compile failure, 2 on spike INCORRECT/Mismatch.

## When NOT to use

- Harness predates `candidate_fn_t` (no typedef). The script aborts — those harnesses use the legacy SUBSTITUTE-HERE inline flow which a full search loop handles.
- User wants real firesim cycle counts. This skill is wrong; run the full search instead.
- User wants to test an LLM-mutated candidate rather than the sol itself. Pass the candidate file as `[sol_path]`.

## Common failure modes

- **Compile error: dtc/west not found** → Zephyr SDK not relocated or zephyr conda env not in PATH for build subprocess (saturn_eval activates it; if that fails, run the SDK installer `.sh` once).
- **Spike timeout (60s)** → candidate has an infinite loop, or sizes are too large for spike. Lower `BATCH_ELEMS` / `KC` in the harness or fix the kernel.
- **`spike: command not found`** → forgot to `source /scratch/kchern2/chipyard/env.sh` before invoking.
