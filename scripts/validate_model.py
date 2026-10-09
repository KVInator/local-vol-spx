"""Check grid refinements for one saved daily surface."""

import argparse
from pathlib import Path

from results import StudyResults
from validation import DailyAHValidator, DailyValidationSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", default="UNKNOWN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = StudyResults(args.study)
    output = saved.safe_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    carry = saved.carry(args.date)
    carry = carry.loc[carry.root.eq(args.root)]
    validator = DailyAHValidator(DailyValidationSettings())
    tables = validator.run(
        saved.model(args.date, args.root),
        saved.calibration_quotes(args.date, args.root),
        carry,
    )
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    from artifacts import atomic_json

    atomic_json(
        output / "audit.json",
        {
            "date": args.date,
            "root": args.root,
            "input_sha256": saved.inputs,
            "tables": list(tables),
        },
    )
    print("Saved validation:", output)


if __name__ == "__main__":
    main()
