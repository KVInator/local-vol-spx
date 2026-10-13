"""Validate a saved AH surface or fit a constrained SSVI quote comparison."""

import argparse
from dataclasses import asdict
from pathlib import Path

from results import StudyResults
from validation import DailyAHValidator, DailyValidationSettings, SSVIValidator
from artifacts import atomic_json, sha256
from research_validation import (
    GreekPanelSettings,
    CalibrationVegaSettings,
    CalibrationVegaValidator,
    GreekPanelWorkflow,
    SavedDecisionRefinement,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--root", default="UNKNOWN")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=["ah", "ssvi", "greek-panel", "calibration-vega", "saved-decisions"],
        default="ah",
    )
    parser.add_argument("--max-evaluations", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--coarse-intervals", type=int, default=12000)
    parser.add_argument("--fine-intervals", type=int, default=24000)
    parser.add_argument("--steps-per-day", type=int)
    parser.add_argument(
        "--radii", type=float, nargs="+", default=[0.0, 0.00025, 0.0005, 0.001]
    )
    parser.add_argument(
        "--iv-bumps", type=float, nargs="+", default=[0.0001, 0.0002, 0.0005]
    )
    args = parser.parse_args()
    saved = StudyResults(args.study)
    output = saved.safe_output(args.output)
    experimental = args.method in ("greek-panel", "calibration-vega", "saved-decisions")
    if experimental and output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output folder for each numerical experiment.")
    # The live index can advance during a validation. Preserve the version read;
    # immutable carry, model and quote files remain checked against their pins.
    index_digest = saved.inputs.pop(str(saved.index_file))
    carry = saved.carry(args.date)
    carry = carry.loc[carry.root.eq(args.root)]
    quotes = saved.calibration_quotes(args.date, args.root)
    settings = None
    if args.method == "greek-panel":
        settings = GreekPanelSettings(
            radii=tuple(args.radii),
            coarse_intervals=args.coarse_intervals,
            fine_intervals=args.fine_intervals,
            steps_per_day=args.steps_per_day or 64,
        )
        validator = GreekPanelWorkflow(settings, args.workers)
        tables = validator.run(saved, args.date, args.root, output)
    elif args.method == "saved-decisions":
        validator = SavedDecisionRefinement()
        values, changes = validator.run(saved, args.date, args.root)
        tables = {
            "saved_decision_repricing": values,
            "saved_decision_refinement": changes,
        }
    elif args.method == "calibration-vega":
        settings = CalibrationVegaSettings(
            bumps=tuple(args.iv_bumps),
            intervals=(args.coarse_intervals, args.fine_intervals),
            steps_per_day=args.steps_per_day or 128,
            radius=saved.freeze["identity"]["config"]["settings"]["panel"]["radius"],
        )
        calibration = saved.freeze["identity"]["config"]["settings"]["calibration"]
        validator = CalibrationVegaValidator(calibration, settings)
        tables = validator.run(saved.model(args.date, args.root), quotes, carry)
    elif args.method == "ssvi":
        validator = SSVIValidator(args.max_evaluations)
        tables = validator.run(quotes, carry)
    else:
        validator = DailyAHValidator(DailyValidationSettings())
        tables = validator.run(saved.model(args.date, args.root), quotes, carry)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "study_index_snapshot.json", saved.index)
    if args.method == "calibration-vega":
        for shock, model in validator.models.items():
            model.save(output / f"ah_iv_shock_{shock:+.8f}.json")
    if args.method == "ssvi":
        validator.surface.save(output / "ssvi_surface.json")
    for name, frame in tables.items():
        if experimental:
            frame = frame.assign(quote_date=args.date, root=args.root)
        frame.to_csv(output / f"{name}.csv", index=False)
    saved.verify_unchanged()

    atomic_json(
        output / "audit.json",
        {
            "date": args.date,
            "root": args.root,
            "method": args.method,
            "status": (
                "completed_with_failures"
                if experimental and len(tables.get("validation_failures", []))
                else "completed"
            ),
            "settings": asdict(settings) if settings is not None else None,
            "source_sha256": {
                str(path.relative_to(Path(__file__).resolve().parents[1])): sha256(path)
                for path in sorted(
                    [
                        *Path(__file__).resolve().parents[1].glob("src/*.py"),
                        *Path(__file__).resolve().parents[1].glob("scripts/*.py"),
                    ]
                )
            },
            "every_entry_greek_independently_validated": False,
            "scope": "Selected numerical experiments, separate from frozen historical outcomes",
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
    if experimental and "validation_failures" in tables:
        failures = tables["validation_failures"]
        print("Recorded validation failures:", len(failures))
        print(
            "Selected contracts:",
            len(
                tables[
                    (
                        "panel_contracts"
                        if args.method == "greek-panel"
                        else "vega_contracts"
                    )
                ]
            ),
        )
        if len(failures):
            print(failures.to_string(index=False))
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
