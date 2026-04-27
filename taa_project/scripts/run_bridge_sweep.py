"""Run a focused bridge sweep around the nested-risk frontier.

Memory rationale:
- Each candidate runs in its own Python process.
- The orchestrator holds only the bridge manifest and run metadata.
- TimesFM forecasts are warmed to disk before the sweep, so bridge runs should
  hit the cache rather than instantiate the model repeatedly.

References:
- Whitmore IPS target search in this repository.
- psutil process memory API:
  https://psutil.readthedocs.io/en/latest/#psutil.Process.memory_info

Point-in-time safety:
- Safe. This script launches isolated backtests and summarizes their outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

from taa_project.analysis.bridge_comparison import BRIDGE_OUTPUT_DIRNAME
from taa_project.config import OUTPUT_DIR
from taa_project.memory import current_rss_gb, guard_process_memory


BRIDGE_RUNS_ROOT = OUTPUT_DIR / "bridge_runs"
BRIDGE_OUTPUT_DIR = OUTPUT_DIR / BRIDGE_OUTPUT_DIRNAME
BRIDGE_MANIFEST_PATH = BRIDGE_OUTPUT_DIR / "bridge_manifest.json"
BRIDGE_RESULTS_PATH = BRIDGE_OUTPUT_DIR / "bridge_sweep_results.json"
SUBPROCESS_RSS_LIMIT_GB = 4.0
ORCHESTRATOR_RSS_LIMIT_GB = 0.2
RUN_TIMEOUT_SECONDS = 3600

BRIDGE_CONFIGS = [
    {
        "run_id": "bridge_nested_base",
        "label": "Nested Base",
        "family": "Nested",
        "args": ["--timesfm", "--nested-risk"],
    },
    {
        "run_id": "bridge_nested_base_bl075",
        "label": "Nested Base + BL 0.75",
        "family": "Nested + BL",
        "args": ["--timesfm", "--nested-risk", "--bl-stress-views", "--bl-stress-shock", "0.75"],
    },
    {
        "run_id": "bridge_nested_core65_sat11",
        "label": "Nested 6.5/11/15",
        "family": "Nested",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.065",
            "--nested-sat-vol",
            "0.11",
            "--nested-nt-vol",
            "0.15",
        ],
    },
    {
        "run_id": "bridge_nested_core65_sat11_bl075",
        "label": "Nested 6.5/11/15 + BL 0.75",
        "family": "Nested + BL",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.065",
            "--nested-sat-vol",
            "0.11",
            "--nested-nt-vol",
            "0.15",
            "--bl-stress-views",
            "--bl-stress-shock",
            "0.75",
        ],
    },
    {
        "run_id": "bridge_nested_core65_sat12",
        "label": "Nested 6.5/12/15",
        "family": "Nested",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.065",
            "--nested-sat-vol",
            "0.12",
            "--nested-nt-vol",
            "0.15",
        ],
    },
    {
        "run_id": "bridge_nested_core65_sat12_bl075",
        "label": "Nested 6.5/12/15 + BL 0.75",
        "family": "Nested + BL",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.065",
            "--nested-sat-vol",
            "0.12",
            "--nested-nt-vol",
            "0.15",
            "--bl-stress-views",
            "--bl-stress-shock",
            "0.75",
        ],
    },
    {
        "run_id": "bridge_nested_core70_sat11",
        "label": "Nested 7/11/15",
        "family": "Nested",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.07",
            "--nested-sat-vol",
            "0.11",
            "--nested-nt-vol",
            "0.15",
        ],
    },
    {
        "run_id": "bridge_nested_core70_sat11_bl075",
        "label": "Nested 7/11/15 + BL 0.75",
        "family": "Nested + BL",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.07",
            "--nested-sat-vol",
            "0.11",
            "--nested-nt-vol",
            "0.15",
            "--bl-stress-views",
            "--bl-stress-shock",
            "0.75",
        ],
    },
    {
        "run_id": "bridge_nested_core65_sat11_shift",
        "label": "Nested 6.5/11/15 Shifted Sleeves",
        "family": "Nested",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.065",
            "--nested-sat-vol",
            "0.11",
            "--nested-nt-vol",
            "0.15",
            "--nested-sleeve-weights",
            "0.53,0.39,0.08",
        ],
    },
    {
        "run_id": "bridge_nested_core65_sat11_shift_bl075",
        "label": "Nested 6.5/11/15 Shifted + BL 0.75",
        "family": "Nested + BL",
        "args": [
            "--timesfm",
            "--nested-risk",
            "--nested-core-vol",
            "0.065",
            "--nested-sat-vol",
            "0.11",
            "--nested-nt-vol",
            "0.15",
            "--nested-sleeve-weights",
            "0.53,0.39,0.08",
            "--bl-stress-views",
            "--bl-stress-shock",
            "0.75",
        ],
    },
]


def _base_env() -> dict[str, str]:
    return {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


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


def _write_manifest() -> None:
    BRIDGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BRIDGE_MANIFEST_PATH.write_text(json.dumps(BRIDGE_CONFIGS, indent=2), encoding="utf-8")


def run_config(run_id: str, args: list[str], start: str, end: str, folds: int) -> dict[str, object]:
    run_root = BRIDGE_RUNS_ROOT / run_id
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
        start,
        "--end",
        end,
        "--folds",
        str(folds),
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

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open("w", encoding="utf-8") as stderr_handle:
        start_time = time.time()
        process = subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle, env=_base_env())
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
    parser = argparse.ArgumentParser(description="Run the focused bridge sweep in isolated subprocesses.")
    parser.add_argument("--start", default="2003-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--skip-warm-cache", action="store_true")
    args = parser.parse_args()

    BRIDGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BRIDGE_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    _write_manifest()

    guard_process_memory("run_bridge_sweep:start", limit_gb=ORCHESTRATOR_RSS_LIMIT_GB)
    if not args.skip_warm_cache:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "taa_project.scripts.warm_timesfm_cache",
                "--start",
                args.start,
                "--end",
                args.end,
                "--folds",
                str(args.folds),
            ],
            check=True,
            env=_base_env(),
        )
        guard_process_memory("run_bridge_sweep:after_warm_cache", limit_gb=ORCHESTRATOR_RSS_LIMIT_GB)

    if BRIDGE_RESULTS_PATH.exists():
        results = json.loads(BRIDGE_RESULTS_PATH.read_text(encoding="utf-8"))
    else:
        results = []
    completed_run_ids = {str(row.get("run_id")) for row in results if int(row.get("exit_code", 1)) == 0}

    for entry in BRIDGE_CONFIGS:
        run_id = str(entry["run_id"])
        if run_id in completed_run_ids:
            print(f"[bridge] skipping {run_id}; already completed")
            continue
        print(f"[bridge] starting {run_id} (rss={current_rss_gb():.3f} GB)")
        result = run_config(run_id=run_id, args=list(entry["args"]), start=args.start, end=args.end, folds=args.folds)
        results.append(result)
        BRIDGE_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"[bridge] finished {run_id} exit={result['exit_code']} "
            f"elapsed={result['elapsed_sec']:.0f}s peak_rss={result['peak_rss_gb']:.3f} GB"
        )
        guard_process_memory(f"run_bridge_sweep:{run_id}:post_run", limit_gb=ORCHESTRATOR_RSS_LIMIT_GB)

    subprocess.run(
        [
            sys.executable,
            "-m",
            "taa_project.scripts.build_bridge_comparison",
            "--run-root",
            str(BRIDGE_RUNS_ROOT),
            "--output-dir",
            str(BRIDGE_OUTPUT_DIR),
            "--manifest-path",
            str(BRIDGE_MANIFEST_PATH),
        ],
        check=True,
        env=_base_env(),
    )
    print(f"Wrote bridge sweep results to {BRIDGE_RESULTS_PATH}")


if __name__ == "__main__":
    main()
