"""Validate a saved AH surface or fit a constrained SSVI quote comparison."""

import argparse
from pathlib import Path

from results import StudyResults
from validation import DailyAHValidator, DailyValidationSettings, SSVIValidator
from artifacts import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", default="UNKNOWN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["ah", "ssvi"], default="ah")
    parser.add_argument("--max-evaluations", type=int, default=400)
    args = parser.parse_args()
    saved = StudyResults(args.study)
    output = saved.safe_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    # The live index can advance during a validation. Preserve the version read;
    # immutable carry, model and quote files remain checked against their pins.
    index_digest = saved.inputs.pop(str(saved.index_file))
    atomic_json(output / "study_index_snapshot.json", saved.index)
    carry = saved.carry(args.date)
    carry = carry.loc[carry.root.eq(args.root)]
    quotes = saved.calibration_quotes(args.date, args.root)
    if args.method == "ssvi":
        validator = SSVIValidator(args.max_evaluations)
        tables = validator.run(quotes, carry)
        validator.surface.save(output / "ssvi_surface.json")
    else:
        validator = DailyAHValidator(DailyValidationSettings())
        tables = validator.run(saved.model(args.date, args.root), quotes, carry)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    saved.verify_unchanged()

    atomic_json(
        output / "audit.json",
        {
            "date": args.date,
            "root": args.root,
            "method": args.method,
            "study_index_at_load_sha256": index_digest,
            "input_sha256": saved.inputs,
            "tables": list(tables),
            "output_sha256": {
                file.name: sha256(file)
                for file in sorted(output.iterdir())
                if file.is_file() and file.name != "audit.json"
            },
            "ssvi_fit": validator.fit_report if args.method == "ssvi" else None,
        },
    )
    print("Saved validation:", output)
    if args.method == "ssvi":
        print("SSVI optimizer converged:", validator.fit_report["optimizer_converged"])
        print(tables["ssvi_quote_fit"].head(1).to_string(index=False))
        print(
            tables["ssvi_shape_checks"]
            .groupby("route")[
                [
                    "negative_calendar_derivatives",
                    "negative_density_samples",
                    "ill_conditioned_denominators",
                ]
            ]
            .sum()
            .to_string()
        )


if __name__ == "__main__":
    main()
