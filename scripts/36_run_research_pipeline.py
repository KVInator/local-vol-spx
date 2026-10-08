"""Verify saved runs or execute a resumable monthly research pipeline."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from research_pipeline import PipelineConfig, ResearchPipeline, STAGES, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", help="Override the output folder; useful for a separate fresh run.")
    parser.add_argument("--ignore-checkpoints", action="store_true",
                        help="Execute fresh stages rather than adopting configured saved runs.")
    parser.add_argument("--plan", action="store_true", help="Print the plan without writing or computing.")
    parser.add_argument("--status", action="store_true", help="Read saved progress without running stages.")
    parser.add_argument("--stop-after", choices=STAGES, default="notebook")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    config = PipelineConfig.load(root / args.config)
    if args.output:
        config = replace(config, output=args.output)
    if args.ignore_checkpoints:
        config = replace(config, checkpoints={})
    if args.plan and args.status:
        parser.error("Choose --plan or --status.")
    if args.status:
        path = (root / config.output) / "pipeline_state.json"
        state = json.loads(path.read_text())
        print("Pipeline status:", state["status"])
        for key, row in state["stages"].items():
            print(f"{key:45} {row['status']:25} {row.get('seconds', 0):9.2f}s")
        return 0
    runner = ResearchPipeline(root, config, args.stop_after)
    if args.plan:
        print(json.dumps(runner.plan(), indent=2))
        print("No files written; no calibration or PDE executed.")
        return 0
    result = runner.run()
    print("\nRun index:", runner.output / "run_index.json")
    print("Progress:", runner.output / "stage_index.csv")
    print("\nMonth coverage:")
    for row in rows(runner.output / "month_summary.csv"):
        print(f"{row['month']}: {row['calibration_dates']} calibration dates, "
              f"{row['entries']} entries, {row['matched_comparisons']} matched comparisons; "
              f"{row['failed_models']} failed models.")
    for month, paths in result.items():
        print(month, paths.get("status", f"stopped after {args.stop_after}"))
        if "notebook" in paths:
            print("Notebook:", paths["notebook"])
    print("Execution completion is not a certification of Greeks, wings or hedge outperformance.")
    return int(runner.store.data["status"] == "completed_with_failures")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
