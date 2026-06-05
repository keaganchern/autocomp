#!/usr/bin/env python3
"""Compile a saturn-rvv test harness and run on spike. No firesim.

Mirrors saturn_eval.py's `_build_and_run_spike` for a single (harness, sol) pair.

Usage:
    python3 run.py <harness_path> [sol_path]

If sol_path is omitted, auto-discovered as sols/<prob_type>/<prob_id>_*.c
where prob_type and prob_id come from the harness path
(harnesses/<prob_type>/test<N>.c).
"""
import re
import sys
import pathlib

SCRIPT = pathlib.Path(__file__).resolve()
# .claude/skills/test-harness/run.py → project root is 3 parents up.
PROJECT_ROOT = SCRIPT.parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from autocomp.backend.saturn.saturn_eval import (
    SaturnTest, SaturnEvalBackend, _build_and_run_spike, clean_code,
    _build_reference_c,
)
from autocomp.search.prob import Prob


def parse_harness_path(harness_path: pathlib.Path) -> tuple[str, int]:
    """harnesses/<prob_type>/test<N>.c → (prob_type, N)."""
    m = re.match(r"test(\d+)\.c$", harness_path.name)
    if not m:
        raise ValueError(f"Harness filename must be test<N>.c, got {harness_path.name!r}")
    return harness_path.parent.name, int(m.group(1))


def find_sol(prob_type: str, prob_id: int) -> pathlib.Path:
    sol_dir = PROJECT_ROOT / "sols" / prob_type
    matches = sorted(sol_dir.glob(f"{prob_id}_*.c"))
    if not matches:
        raise FileNotFoundError(f"No sol matching {prob_id}_*.c in {sol_dir}")
    return matches[0]


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__, file=sys.stderr)
        return 64
    harness = pathlib.Path(argv[1]).resolve()
    if not harness.exists():
        print(f"Harness not found: {harness}", file=sys.stderr)
        return 66

    prob_type, prob_id = parse_harness_path(harness)
    sol = pathlib.Path(argv[2]).resolve() if len(argv) > 2 else find_sol(prob_type, prob_id)
    if not sol.exists():
        print(f"Sol not found: {sol}", file=sys.stderr)
        return 66

    print(f"[harness]   {harness}")
    print(f"[sol]       {sol}")
    print(f"[prob_type] {prob_type}")
    print(f"[prob_id]   {prob_id}")

    sol_text = sol.read_text()
    sol_body = clean_code(sol_text)
    params = SaturnEvalBackend._extract_candidate_params(sol_text)
    param_sig = ", ".join(f"{t} {n}" for t, n in params) if params else "void"

    test = SaturnTest(str(harness))
    if not test.uses_candidate_fn_style():
        print(
            "ERROR: harness has no candidate_fn_t typedef — legacy SUBSTITUTE-HERE "
            "inline harnesses are not supported by this skill.",
            file=sys.stderr,
        )
        return 65

    injected_main_c = test.inject_candidates([sol_body], [0], param_sig)

    print(f"[param_sig] {param_sig}")
    print(f"[building]  slot 0 ... (this is the same path saturn_eval uses)")
    print("=" * 70)

    # Build reference.c the same way the real evaluator does so tier-2
    # harnesses (extern reference_kernel) work under the skill too.
    prob = Prob(prob_type, prob_id)
    ref_c = _build_reference_c(prob)
    _idx, output = _build_and_run_spike((injected_main_c, 0, 0, prob_type, ref_c))
    print(output)
    print("=" * 70)

    if output.startswith("Compile error") or output.startswith("Compile timeout"):
        print("[FAIL] compile", file=sys.stderr)
        return 1
    if output.startswith("Binary not found") or output.startswith("Build error"):
        print("[FAIL] build", file=sys.stderr)
        return 1
    if output.startswith("Timeout") or output.startswith("Spike error"):
        print("[FAIL] spike runtime", file=sys.stderr)
        return 3
    if "INCORRECT" in output or "Mismatch" in output:
        print("[FAIL] correctness", file=sys.stderr)
        return 2
    if "Correct result" in output:
        print("[PASS] compile + spike + correctness")
        return 0
    # Spike ran but emitted neither marker — surface the output as ambiguous.
    print("[WARN] spike ran but no Correct/INCORRECT marker in output", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
