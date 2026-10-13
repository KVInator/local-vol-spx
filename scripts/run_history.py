"""Run or resume a historical study."""

import argparse
import json
from pathlib import Path

from artifacts import study_lock
from study import HistoricalStudy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config/history_2013_2023.json")
    )
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--plan", action="store_true")
    controls.add_argument("--prepare-calendar", action="store_true")
    controls.add_argument("--freeze", action="store_true")
    parser.add_argument("--until-date")
    args = parser.parse_args()
    if args.until_date and (args.plan or args.prepare_calendar or args.freeze):
        parser.error("Use --until-date only when running a study.")
    study = HistoricalStudy(args.config, Path(__file__).resolve().parents[1])
    if args.plan:
        print(json.dumps(study.plan(), indent=2))
    elif args.prepare_calendar:
        print("Calendar:", study.prepare_calendar())
    elif args.freeze:
        with study_lock(study.output):
            print("Study fingerprint:", study.freeze()["fingerprint"])
    else:
        timing = study.run(args.until_date)
        print(json.dumps(timing, indent=2))
        print("Saved results:", study.output)


if __name__ == "__main__":
    main()
