"""Trace saved calibration outliers and selected contracts without rerunning models."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd

import pilot_result_audit
from pilot_result_audit import PilotResultAudit, ResultAuditReport, ResultAuditSettings, SavedResearchRun, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=Path,
        default=Path("outputs/research_pipeline/october_full_v1/run_index.json"))
    parser.add_argument("--month", default="2023-10")
    parser.add_argument("--focus-dates", nargs="+", default=["2023-10-19", "2023-10-26", "2023-10-27"])
    parser.add_argument("--selection-dates", nargs="+", default=["2023-10-11"])
    parser.add_argument("--output", type=Path, default=Path("outputs/pilot_result_audit/october_2023"))
    parser.add_argument("--plan", action="store_true", help="Verify the explicit saved inputs; write nothing.")
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    settings = ResultAuditSettings(tuple(args.focus_dates), tuple(args.selection_dates))
    run = SavedResearchRun(args.run_index, args.month).load()
    parent = run.safe_output(args.output)
    if args.plan:
        print(json.dumps({"month": args.month, "stages": {key: str(value) for key, value in run.folders.items()},
            "focus_dates": args.focus_dates, "selection_dates": args.selection_dates,
            "verified_input_files": len(run.inputs), "output_parent": str(parent),
            "model_or_PDE_execution": False}, indent=2))
        print("No files written; no calibration, pricing, PDE or hedge ledger executed.")
        return
    tables, summary = PilotResultAudit(settings).run(run.frames, run.audits["panel"]["settings"],
        run.audits["carry"]["settings"]["parity_window"])
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output = parent / stamp
    output.mkdir(parents=True, exist_ok=False)
    audit = {"status": "running", "month": args.month, "settings": asdict(settings),
             "upstream_run_index": str(run.index_file), "started_utc": started_utc}
    audit_file = output / "audit.json"
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    try:
        for name, frame in tables.items():
            frame.to_csv(output / f"{name}.csv", index=False)
        report = ResultAuditReport(tables, settings)
        report.plots(output / "plots")
        report.notebook(output / "result_audit.ipynb")
        run.verify_unchanged()
        audit.update(status="completed", summary=summary, input_sha256=run.inputs,
            upstream_stages={key: str(value) for key, value in run.folders.items()},
            pricing_carry_case=run.audits["panel"].get("carry_case"),
            saved_panel_settings=run.audits["panel"]["settings"],
            source_sha256={str(path.resolve()): sha256(path) for path in
                          [Path(__file__), Path(pilot_result_audit.__file__)]},
            output_sha256={path.relative_to(output).as_posix(): sha256(path)
                           for path in sorted(output.rglob("*")) if path.is_file() and path != audit_file},
            band_comparison="Each selected contract's own observed call or put bid/ask, tolerance 1e-6 index points.",
            original_price="Saved AH calibration-strike call price, converted to put by the pinned same-date carry when needed.",
            smoothed_price="Saved ready backward-panel price; not recomputed by this audit.",
            missing_price_policy="Retained with explicit status; no interpolation, model load or price substitution.",
            price_change_scope="Smoothed minus original includes a changed diffusion and numerical error.",
            parity_scope="Same-date observed call/put pairs; prepared forward and assumed discount remain fixed.",
            focus_dates_scope="Retrospective diagnostic views only; all original comparisons are retained.",
            performance_scope="Saved development-sample P&L; date-level associations do not establish causation.",
            native_grid_refinement_performed=False, all_entry_greeks_independently_validated=False,
            contract_identity_verified=False, snapshot_provenance_verified=False,
            funding_curve_verified=False, wing_robustness_certified=False,
            models_loaded=False, models_refitted=False, pde_rerun=False, hedge_ledger_rerun=False,
            deltas_changed=False, observations_modified=False, prices_clipped=False,
            entry_rules_changed=False, comparisons_removed=False, candidate_promoted=False,
            out_of_sample_claim=False, executable_hedge_returns_claimed=False,
            versions={"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
            finished_utc=datetime.now(timezone.utc).isoformat(),
            elapsed_wall_seconds=time.perf_counter()-started,
            timing_scope="Input verification, table audit, figures and notebook generation through output hashing; final audit write excluded.")
    except BaseException as error:
        audit.update(status="failed", message=f"{type(error).__name__}: {error}",
                     elapsed_wall_seconds=time.perf_counter()-started)
        audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        raise
    audit_file.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Saved-results audit; no model or PDE executed.")
    print("\nMonth summary:\n" + pd.DataFrame([summary]).to_string(index=False))
    columns = ["quote_date", "root", "calibration_quotes", "rms_half_spreads", "outside_original_bands",
        "parity_incompatible_groups", "fitted_parity_outside_groups", "original_outside_entry_bands",
        "smoothed_outside_entry_bands"]
    print("\nFocus-date diagnostics:\n" + tables["daily_audit"].loc[lambda f: f.focus_date, columns].to_string(index=False))
    print("\nSelection-date diagnostics:\n" + tables["selection_daily"].loc[lambda f: f.selection_focus_date].to_string(index=False))
    print(f"\nMeasured audit wall time: {audit['elapsed_wall_seconds']:.3f} seconds.")
    print(f"Audit: {output}")
    print(f"Notebook: {output / 'result_audit.ipynb'}")
    print("Four figures saved. All entries retained; diagnostic associations are not causal conclusions.")


if __name__ == "__main__":
    main()
