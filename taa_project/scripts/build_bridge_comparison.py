"""Build comparison and ranking artifacts for the focused bridge sweep.

References:
- Whitmore IPS bridge-search workflow in this repository.

Point-in-time safety:
- Safe. This script reads already-generated OOS artifacts only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from taa_project.analysis.bridge_comparison import (
    BRIDGE_COMPARISON_FILENAME,
    BRIDGE_OUTPUT_DIRNAME,
    BRIDGE_RANKING_FILENAME,
    build_bridge_comparison,
    rank_bridge_candidates,
    select_bridge_candidate,
    write_bridge_selection,
)
from taa_project.config import OUTPUT_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Build comparison and ranking outputs for the bridge sweep.")
    parser.add_argument("--run-root", default=str(OUTPUT_DIR / "bridge_runs"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR / BRIDGE_OUTPUT_DIRNAME))
    parser.add_argument("--manifest-path", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path) if args.manifest_path else None
    comparison = build_bridge_comparison(
        run_root=Path(args.run_root),
        output_dir=Path(args.output_dir),
        manifest_path=manifest_path,
    )
    ranked = rank_bridge_candidates(comparison)
    ranking_path = Path(args.output_dir) / BRIDGE_RANKING_FILENAME
    ranked.to_csv(ranking_path, index=False)
    selection = select_bridge_candidate(ranked)
    selection_path = write_bridge_selection(selection, output_dir=Path(args.output_dir))
    print(
        f"Wrote {Path(args.output_dir) / BRIDGE_COMPARISON_FILENAME}, "
        f"{ranking_path}, and {selection_path}."
    )


if __name__ == "__main__":
    main()
