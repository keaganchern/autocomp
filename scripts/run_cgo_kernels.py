#!/usr/bin/env python3
"""Sequentially run autocomp on a range of cgo-kernels prob_ids.

Wraps `python -m autocomp.search.run_search` with the AUTOCOMP_PROB_ID env
var set per iteration, captures per-prob stdout+stderr to a log file, and
continues on failure so a single bad kernel doesn't kill the sweep.

Defaults to prob_ids 1..6 (the cgo-kernels harnesses created alongside
test0). Override with --probs or --range.

Usage examples:
    python scripts/run_cgo_kernels.py                          # 1..6
    python scripts/run_cgo_kernels.py --range 0 6              # 0..6 (all)
    python scripts/run_cgo_kernels.py --probs 2 4 5            # explicit set
    python scripts/run_cgo_kernels.py --probs 3 --no-clean     # skip build-slot wipe
"""
from __future__ import annotations

import argparse
import atexit
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

REPO_ROOT       = pathlib.Path(__file__).resolve().parent.parent
SLOTS_DIR       = REPO_ROOT / "autocomp" / "backend" / "saturn" / "tmp_dir"
LOG_DIR         = REPO_ROOT / "cgo_kernels_logs"
SOLS_DIR        = REPO_ROOT / "sols" / "cgo-kernels"
HARNESSES_DIR   = REPO_ROOT / "harnesses" / "cgo-kernels"

# Same constants saturn_eval.py uses, kept in sync by hand. If the chipyard
# install moves, update both files.
CHIPYARD_PATH   = pathlib.Path("/scratch/kchern2/chipyard")
FIRESIM_PATH    = CHIPYARD_PATH / "sims" / "firesim"
FIRESIM_KILL_TIMEOUT = 180   # max wall-clock for `firesim kill` to finish

# Track the active child PID so the atexit handler can guarantee firesim is
# torn down even if we're killed by an uncaught exception or hard signal that
# bypassed run_one()'s normal cleanup path.
_active_child_pgid: int | None = None
_firesim_killed_this_run = False


def firesim_kill(reason: str) -> None:
    """Best-effort `firesim kill`. Idempotent — safe to call multiple times
    and from atexit. Always exits quickly: bounded timeout, swallows all
    errors. This is the safety net that protects the server when an
    interrupt arrives mid-`infrasetup` or mid-`runworkload`."""
    global _firesim_killed_this_run
    if _firesim_killed_this_run:
        return
    if not FIRESIM_PATH.exists():
        return
    print(f"[wrapper] firesim kill ({reason})…", flush=True)
    cmd = (f"cd {FIRESIM_PATH} && source {FIRESIM_PATH}/sourceme-manager.sh "
           f"&& firesim kill")
    try:
        subprocess.run(
            ["bash", "-c", cmd],
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            timeout=FIRESIM_KILL_TIMEOUT,
        )
        _firesim_killed_this_run = True
    except subprocess.TimeoutExpired:
        print(f"[wrapper] firesim kill itself timed out after "
              f"{FIRESIM_KILL_TIMEOUT}s — server may need manual cleanup",
              file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[wrapper] firesim kill error (continuing): {e}",
              file=sys.stderr, flush=True)


def _atexit_cleanup() -> None:
    """Last-resort cleanup. Runs on normal exit, sys.exit(), and most uncaught
    exceptions. NOT called on SIGKILL — nothing can save us from that."""
    if _active_child_pgid is not None:
        # Child still tracked → kill the whole process group then firesim.
        try:
            os.killpg(_active_child_pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    firesim_kill("atexit")


atexit.register(_atexit_cleanup)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--probs", type=int, nargs="+",
                     help="Explicit prob_id list (e.g. --probs 2 4 5).")
    sel.add_argument("--range", type=int, nargs=2, metavar=("LO", "HI"),
                     help="Inclusive range, e.g. --range 1 6.")
    p.add_argument("--no-clean", action="store_true",
                   help="Skip wiping build_slot_* before the sweep. "
                        "By default slots are cleared once at the start so the "
                        "latest harness_common.h + reference.c preamble propagate.")
    return p.parse_args()


def resolve_probs(args: argparse.Namespace) -> list[int]:
    if args.probs:
        return sorted(set(args.probs))
    if args.range:
        lo, hi = args.range
        return list(range(lo, hi + 1))
    return list(range(1, 7))   # default: the newly created harnesses


def precheck(probs: list[int]) -> list[int]:
    """Skip prob_ids that have no sol or no harness on disk."""
    ok: list[int] = []
    for pid in probs:
        sol = list(SOLS_DIR.glob(f"{pid}_*.c"))
        harness = HARNESSES_DIR / f"test{pid}.c"
        if not sol:
            print(f"  [SKIP] prob {pid}: no sol in {SOLS_DIR}", file=sys.stderr)
            continue
        if not harness.exists():
            print(f"  [SKIP] prob {pid}: no harness at {harness}", file=sys.stderr)
            continue
        ok.append(pid)
    return ok


def clean_build_slots() -> int:
    """Wipe every build_slot_* under saturn tmp_dir. Return count cleared."""
    if not SLOTS_DIR.exists():
        return 0
    cleared = 0
    for slot in SLOTS_DIR.glob("build_slot_*"):
        shutil.rmtree(slot, ignore_errors=True)
        cleared += 1
    return cleared


def _graceful_kill(pgid: int, label: str) -> None:
    """Two-stage process-group shutdown:
       1. SIGINT — gives Python's `finally` blocks a chance to run, which
          includes saturn_eval's `firesim kill` + 120s drain in
          run_firesim_batch.
       2. SIGTERM after 240s if still alive.
       3. SIGKILL after another 30s as last resort.
    Then `firesim kill` is invoked unconditionally as a safety net (see
    docstring of firesim_kill — idempotent)."""
    def _send(sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _alive() -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    print(f"\n[wrapper] interrupting prob {label}: SIGINT → process group {pgid}",
          flush=True)
    if not _send(signal.SIGINT):
        return

    # Saturn_eval's finally block can take up to 120s (firesim kill timeout)
    # + 120s (proc.wait after kill). Give it the full budget plus margin.
    for _ in range(240):
        time.sleep(1)
        if not _alive():
            break
    else:
        print(f"[wrapper] still alive after SIGINT, escalating to SIGTERM",
              flush=True)
        _send(signal.SIGTERM)
        for _ in range(30):
            time.sleep(1)
            if not _alive():
                break
        else:
            print(f"[wrapper] still alive after SIGTERM, SIGKILL", flush=True)
            _send(signal.SIGKILL)
    # Belt-and-suspenders — saturn_eval's own firesim kill may not have run.
    firesim_kill(f"after interrupt of prob {label}")


def run_one(pid: int) -> tuple[int, float]:
    """Run autocomp for one prob_id. Tees output to stdout AND a log file.
    Returns (exit_code, wall_seconds). On KeyboardInterrupt, escalates
    SIGINT → SIGTERM → SIGKILL across the whole child process group, then
    invokes `firesim kill` as a safety net and re-raises."""
    global _active_child_pgid, _firesim_killed_this_run

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"prob{pid}-{ts}.log"

    env = os.environ.copy()
    env["AUTOCOMP_PROB_ID"] = str(pid)
    # Force unbuffered Python stdout in the child so its lines appear live.
    env["PYTHONUNBUFFERED"] = "1"

    # Reset firesim-killed flag for this prob.
    _firesim_killed_this_run = False

    print(f"\n=== prob {pid} starting (log: {log_path}) ===", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "autocomp.search.run_search"],
        cwd=REPO_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,           # line-buffered
        start_new_session=True,          # → new process group; SIGINT goes to entire tree
    )
    _active_child_pgid = os.getpgid(proc.pid)
    interrupted = False
    try:
        with open(log_path, "w") as logf:
            assert proc.stdout is not None
            try:
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    logf.write(line)
                    logf.flush()
            except KeyboardInterrupt:
                interrupted = True
                _graceful_kill(_active_child_pgid, str(pid))
                raise
        rc = proc.wait()
    finally:
        _active_child_pgid = None
        # If we didn't interrupt and saturn_eval already cleaned up firesim
        # (normal exit), skip the extra `firesim kill` to avoid wasting time.
        # On any non-zero exit code we run it defensively — the child may
        # have crashed before reaching its own cleanup code.
        if not interrupted and proc.returncode not in (0, None):
            firesim_kill(f"prob {pid} exited {proc.returncode}")
    elapsed = time.time() - t0
    status = "OK" if rc == 0 else f"FAIL ({rc})"
    print(f"=== prob {pid} done in {elapsed:.0f}s: {status} ===", flush=True)
    return rc, elapsed


def main() -> int:
    args = parse_args()
    probs = resolve_probs(args)
    if not probs:
        print("No prob_ids selected.", file=sys.stderr)
        return 2

    probs = precheck(probs)
    if not probs:
        print("No runnable prob_ids after sol/harness precheck.", file=sys.stderr)
        return 2

    print(f"Sweeping cgo-kernels probs: {probs}")
    if not args.no_clean:
        n = clean_build_slots()
        print(f"Cleared {n} build slot(s) so latest header/preamble changes propagate.")

    results: list[tuple[int, int, float]] = []
    sweep_start = time.time()
    for pid in probs:
        try:
            rc, secs = run_one(pid)
            results.append((pid, rc, secs))
        except KeyboardInterrupt:
            print(f"\nInterrupted at prob {pid}.", file=sys.stderr)
            break
        except Exception as e:  # noqa: BLE001 — never abort the sweep
            print(f"  prob {pid}: launcher error: {e}", file=sys.stderr)
            results.append((pid, -1, 0.0))

    total = time.time() - sweep_start
    print(f"\n=== Sweep complete in {total:.0f}s ===")
    print(f"{'prob':>4}  {'status':>10}  {'secs':>8}")
    for pid, rc, secs in results:
        status = "OK" if rc == 0 else ("ERROR" if rc < 0 else f"EXIT {rc}")
        print(f"{pid:>4}  {status:>10}  {secs:>8.0f}")

    return 0 if all(rc == 0 for _, rc, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
