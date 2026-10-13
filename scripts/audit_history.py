"""Inspect carry exclusions and replay selected scalar ledgers from saved cache."""
import argparse
from pathlib import Path

from artifacts import atomic_json, sha256
from results import StudyResults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dates",
        nargs="+",
        default=[
            "2020-03-02",
            "2020-03-03",
            "2020-03-16",
            "2020-03-26",
            "2023-09-19",
            "2023-11-09",
        ],
    )
    parser.add_argument(
        "--strategies", nargs="+", default=["ah_pde", "surface_sticky_delta"]
    )
    args = parser.parse_args()
    saved = StudyResults(args.study)
    if saved.index.get("status") != "execution_complete":
        raise ValueError("This review requires a completed historical study.")
    output = saved.safe_output(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty review folder.")
    tables = {
        "carry_exclusions": saved.carry_exclusions(),
        **saved.replay_tail_paths(args.dates, strategies=args.strategies),
    }
    saved.verify_unchanged()
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    atomic_json(
        output / "audit.json",
        dict(
            dates=args.dates,
            strategies=args.strategies,
            scope="Recorded carry failures and outcome-selected five-session ledger diagnostics",
            frozen_fingerprint=saved.freeze["fingerprint"],
            input_sha256=saved.inputs,
            output_sha256={p.name: sha256(p) for p in output.glob("*.csv")},
        ),
    )
    print("Saved cache review:", output)
    print(
        "Excluded carry dates:",
        tables["carry_exclusions"].quote_date.nunique()
        if not tables["carry_exclusions"].empty
        else 0,
    )
    print("Compared ledger scenarios:", len(tables["tail_ledger_summary"]))


if __name__ == "__main__":
    main()
