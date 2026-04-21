"""Run the canonical Whitmore configuration sweep in isolated subprocesses.

Each configuration runs in its own Python process so TimesFM, pandas, and
cvxpy memory is fully released between runs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from taa_project.config import FIGURES_DIR, NOTEBOOK_DIR, OUTPUT_DIR, REPORT_DIR
from taa_project.memory import current_rss_gb, guard_process_memory


RUNS_ROOT = OUTPUT_DIR / "runs"
SWEEP_RESULTS_PATH = OUTPUT_DIR / "sweep_results.json"
SUBPROCESS_RSS_LIMIT_GB = 4.0
ORCHESTRATOR_RSS_LIMIT_GB = 0.2
RUN_TIMEOUT_SECONDS = 3600

CONFIGS = [
    ("baseline", ["--no-timesfm", "--vol-budget", "0.10"]),
    ("timesfm_vb10", ["--timesfm", "--vol-budget", "0.10"]),
    ("timesfm_vb08", ["--timesfm", "--vol-budget", "0.08"]),
    ("timesfm_vb07", ["--timesfm", "--vol-budget", "0.07"]),
    ("timesfm_regime_vb", ["--timesfm", "--regime-vol-budgets", '{"risk_on":0.10,"neutral":0.08,"stress":0.05}']),
    ("timesfm_regime_dd", ["--timesfm", "--regime-vol-budgets", '{"risk_on":0.10,"neutral":0.08,"stress":0.05}', "--dd-guardrail"]),
    ("cvar95_vb_2_5", ["--timesfm", "--optimizer-mode", "cvar", "--cvar-alpha", "0.95", "--cvar-budget", "0.025"]),
    ("cvar99_vb_4_0", ["--timesfm", "--optimizer-mode", "cvar", "--cvar-alpha", "0.99", "--cvar-budget", "0.040"]),
    ("nested_risk_default", ["--timesfm", "--nested-risk"]),
    ("nested_risk_cvar", ["--timesfm", "--nested-risk", "--optimizer-mode", "cvar", "--cvar-alpha", "0.95", "--cvar-budget", "0.025"]),
    ("hrp_saa", ["--timesfm", "--vol-budget", "0.10", "--saa-method", "hrp"]),
    ("bl_stress_full", ["--timesfm", "--vol-budget", "0.10", "--bl-stress-views", "--bl-stress-shock", "1.0"]),
    (
        "kitchen_sink",
        [
            "--timesfm",
            "--optimizer-mode",
            "cvar",
            "--cvar-alpha",
            "0.95",
            "--cvar-budget",
            "0.025",
            "--nested-risk",
            "--saa-method",
            "hrp",
            "--bl-stress-views",
            "--regime-vol-budgets",
            '{"risk_on":0.10,"neutral":0.08,"stress":0.05}',
            "--dd-guardrail",
        ],
    ),
]


def _tail_text(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]


def _process_tree_rss_gb(pid: int) -> float:
    try:
        process = psutil.Process(pid)
    except (psutil.Error, ProcessLookupError):
        return 0.0
    try:
        rss = float(process.memory_info().rss)
    except (psutil.Error, ProcessLookupError):
        return 0.0
    try:
        children = process.children(recursive=True)
    except (psutil.Error, PermissionError):
        children = []
    for child in children:
        try:
            rss += float(child.memory_info().rss)
        except (psutil.Error, PermissionError):
            continue
    return rss / 1024**3


def run_config(run_id: str, args: list[str]) -> dict[str, object]:
    run_root = RUNS_ROOT / run_id
    output_dir = run_root / "outputs"
    figure_dir = run_root / "figures"
    report_dir = run_root / "reports"
    notebook_dir = run_root / "notebooks"
    log_dir = run_root / "logs"
    for path in (output_dir, figure_dir, report_dir, notebook_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    stdout_path = log_dir / "stdout.log"
    stderr_path = log_dir / "stderr.log"
    cmd = [
        sys.executable,
        "-m",
        "taa_project.main",
        "--start",
        "2003-01-01",
        "--end",
        "2025-12-31",
        "--folds",
        "5",
        "--output-dir",
        str(output_dir),
        "--figure-dir",
        str(figure_dir),
        "--report-dir",
        str(report_dir),
        "--notebook-dir",
        str(notebook_dir),
        *args,
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle, env=env)
        peak_rss_gb = 0.0
        timed_out = False

        while True:
            return_code = process.poll()
            peak_rss_gb = max(peak_rss_gb, _process_tree_rss_gb(process.pid))
            if peak_rss_gb > SUBPROCESS_RSS_LIMIT_GB:
                process.kill()
                raise MemoryError(
                    f"Run {run_id} exceeded the {SUBPROCESS_RSS_LIMIT_GB:.1f} GB RSS limit: {peak_rss_gb:.3f} GB"
                )
            if return_code is not None:
                break
            if time.time() - start_time > RUN_TIMEOUT_SECONDS:
                timed_out = True
                process.kill()
                return_code = process.wait()
                break
            time.sleep(1.0)

    elapsed_sec = time.time() - start_time
    result = {
        "run_id": run_id,
        "exit_code": int(return_code),
        "elapsed_sec": float(elapsed_sec),
        "peak_rss_gb": float(peak_rss_gb),
        "timed_out": bool(timed_out),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "stderr_tail": _tail_text(stderr_path),
    }
    if timed_out:
        raise TimeoutError(f"Run {run_id} hit the {RUN_TIMEOUT_SECONDS} second timeout.")
    if result["exit_code"] != 0:
        raise RuntimeError(f"Run {run_id} failed with exit code {result['exit_code']}.")
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    guard_process_memory("run_sweep:start", limit_gb=ORCHESTRATOR_RSS_LIMIT_GB)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "taa_project.scripts.warm_timesfm_cache",
            "--start",
            "2003-01-01",
            "--end",
            "2025-12-31",
            "--folds",
            "5",
        ],
        check=True,
    )
    guard_process_memory("run_sweep:after_warm_cache", limit_gb=ORCHESTRATOR_RSS_LIMIT_GB)

    if SWEEP_RESULTS_PATH.exists():
        results = json.loads(SWEEP_RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        results = []
    completed_run_ids = {str(row.get("run_id")) for row in results if int(row.get("exit_code", 1)) == 0}
    for run_id, args in CONFIGS:
        if run_id in completed_run_ids:
            print(f"[orchestrator] skipping {run_id}; already completed")
            continue
        print(f"[orchestrator] starting {run_id} (rss={current_rss_gb():.3f} GB)")
        result = run_config(run_id, args)
        results.append(result)
        SWEEP_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"[orchestrator] finished {run_id} exit={result['exit_code']} "
            f"elapsed={result['elapsed_sec']:.0f}s peak_rss={result['peak_rss_gb']:.3f} GB"
        )
        guard_process_memory(f"run_sweep:{run_id}:post_run", limit_gb=ORCHESTRATOR_RSS_LIMIT_GB)

    subprocess.run([sys.executable, "-m", "taa_project.scripts.build_comparison"], check=True)
    subprocess.run([sys.executable, "-m", "taa_project.scripts.build_submission"], check=True)
    print(f"Wrote sweep results to {SWEEP_RESULTS_PATH}")


if __name__ == "__main__":
    main()
