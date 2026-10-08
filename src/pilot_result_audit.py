"""Audit saved calibration, carry, contract selection and hedge results.

No pricing model, optimizer, PDE solver or hedge ledger is imported or run.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


GROUP = ["quote_date", "root", "expire_date"]
STRIKE_KEY = GROUP + ["strike"]
SIDE_KEY = STRIKE_KEY + ["kind"]
DTYPES = {key: str for key in ["quote_date", "root", "expire_date", "entry_id",
                              "contract_id", "kind", "source_kind", "case"]}


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(frame, columns, label, keys=None):
    missing = sorted(set(columns) - set(frame.columns))
    if missing or frame.columns.has_duplicates:
        raise ValueError(f"{label}: missing or duplicate columns: {missing}")
    if keys and (frame[keys].isna().any().any() or frame.duplicated(keys).any()):
        raise ValueError(f"{label}: missing or duplicate keys {keys}")


def boolean(series):
    values = series.map({True: True, False: False, "True": True, "False": False})
    if values.isna().any():
        raise ValueError(f"Invalid boolean values: {series.name}")
    return values.astype(bool)


def close(actual, expected, label, atol=1e-8):
    if not np.allclose(actual, expected, rtol=1e-11, atol=atol, equal_nan=False):
        raise ValueError(f"Reconciliation failed: {label}")


@dataclass(frozen=True)
class ResultAuditSettings:
    focus_dates: tuple = ("2023-10-19", "2023-10-26", "2023-10-27")
    selection_dates: tuple = ("2023-10-11",)
    quote_band_tolerance: float = 1e-6

    def __post_init__(self):
        for values in (self.focus_dates, self.selection_dates):
            if len(set(values)) != len(values):
                raise ValueError("Duplicate diagnostic dates")
            for date in values:
                if pd.Timestamp(date).strftime("%Y-%m-%d") != date:
                    raise ValueError("Diagnostic dates must use YYYY-MM-DD")
        if not np.isfinite(self.quote_band_tolerance) or self.quote_band_tolerance < 0:
            raise ValueError("Invalid quote-band tolerance")


class SavedResearchRun:
    """Resolve an explicit run index and verify every consumed producer file."""

    FILES = {
        "prepare": ("pilot_quotes",),
        "carry": ("primary_carry", "calibration_quotes"),
        "calibration": ("model_manifest", "quote_residuals", "expiry_summary"),
        "panel": ("delta_panel", "selection_buckets"),
        "comparison": ("paired_results", "coverage"),
        "attribution": ("attribution", "daily_attribution", "group_summary"),
    }

    def __init__(self, index_file, month, repository=None):
        self.index_file = Path(index_file).resolve()
        self.repository = Path(repository or Path.cwd()).resolve()
        self.month = month
        self.inputs, self.audits, self.folders, self.frames = {}, {}, {}, {}
        self.index = self.read_json(self.index_file)
        if month not in self.index.get("months", {}):
            raise ValueError(f"Month {month} is absent from the explicit run index")
        self.record = self.index["months"][month]

    def read_json(self, path):
        self.inputs[str(path)] = sha256(path)
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def resolve(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.repository / path).resolve()

    def read_table(self, folder, audit, name, legacy_input_pins=()):
        path = (folder / f"{name}.csv").resolve()
        if not path.is_relative_to(folder):
            raise ValueError("Producer file escapes its run folder")
        actual = sha256(path)
        expected = audit.get("output_sha256", {}).get(f"{name}.csv")
        # Script 26's legacy preparation audit has no output hashes. Its quote
        # bytes must instead match the already-saved carry producer's input pin.
        if expected != actual and not (expected is None and actual in legacy_input_pins):
            raise ValueError(f"Checksum mismatch or missing output pin: {path}")
        self.inputs[str(path)] = actual
        return pd.read_csv(path, dtype=DTYPES)

    def check_link(self, consumer, producer, name):
        path = self.folders[producer] / name
        digest = self.inputs[str(path.resolve())]
        if digest not in self.audits[consumer].get("input_sha256", {}).values():
            raise ValueError(f"{consumer} does not pin {producer}/{name}")

    def load(self):
        for stage, names in self.FILES.items():
            if stage not in self.record:
                raise ValueError(f"Run index is missing stage {stage}")
            folder = self.resolve(self.record[stage])
            audit = self.read_json(folder / "audit.json")
            self.folders[stage], self.audits[stage] = folder, audit
        for stage, names in self.FILES.items():
            folder, audit = self.folders[stage], self.audits[stage]
            for name in names:
                pins = self.audits["carry"].get("input_sha256", {}).values() if stage == "prepare" else ()
                self.frames[name] = self.read_table(folder, audit, name, pins)
        for consumer, producer, names in (
            ("carry", "prepare", ["pilot_quotes.csv"]),
            ("calibration", "carry", ["calibration_quotes.csv", "primary_carry.csv"]),
            ("panel", "calibration", ["audit.json", "model_manifest.csv"]),
            ("panel", "carry", ["primary_carry.csv"]),
            ("comparison", "panel", ["delta_panel.csv"]),
            ("attribution", "comparison", ["paired_results.csv", "coverage.csv"]),
        ):
            for name in names:
                self.check_link(consumer, producer, name)
        if self.audits["panel"].get("marks_replaced_by_model_prices") is not False:
            raise ValueError("This audit requires observed hedge-panel marks")
        if self.audits["panel"].get("entries_filtered_by_future_availability") is not False:
            raise ValueError("Unexpected future-availability entry screening")
        self.load_validation()
        self.verify_unchanged()
        return self

    def load_validation(self):
        requested = set(self.record.get("validation_scope", []))
        index = self.index_file.parent / "stage_index.csv"
        validation = {}
        if index.exists():
            self.inputs[str(index)] = sha256(index)
            stages = pd.read_csv(index, dtype=str).fillna("")
            require(stages, ["job", "status", "folder"], "stage index", ["job"])
            prefix = f"{self.month}/validation/"
            for row in stages.loc[stages.job.str.startswith(prefix)].to_dict("records"):
                date = row["job"].removeprefix(prefix)
                validation[date] = row
        pieces = {name: [] for name in ["quote_fit", "sensitivity", "forward_shapes", "forward_backward"]}
        scope = []
        dates = sorted(set(self.frames["model_manifest"].quote_date) | requested | set(validation))
        for date in dates:
            row = validation.get(date, {})
            status = row.get("status", "requested_job_missing" if date in requested else "not_requested")
            item = dict(quote_date=date, requested=date in requested, stage_status=status,
                        diagnostics_loaded=False, all_entry_greeks_independently_validated=False)
            if status in {"completed", "completed_with_failures"} and row.get("folder"):
                folder = self.resolve(row["folder"])
                audit = self.read_json(folder / "audit.json")
                if self.inputs[str(self.folders["calibration"] / "audit.json")] not in audit.get("input_sha256", {}).values():
                    raise ValueError(f"Validation {date} does not pin this calibration run")
                for name in pieces:
                    frame = self.read_table(folder, audit, name)
                    if "quote_date" not in frame or not frame.quote_date.eq(date).all():
                        raise ValueError(f"Validation job/date mismatch: {date}/{name}")
                    pieces[name].append(frame)
                item["diagnostics_loaded"] = True
            scope.append(item)
        self.frames["validation_coverage"] = pd.DataFrame(scope)
        for name, parts in pieces.items():
            self.frames[f"validation_{name}"] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    def verify_unchanged(self):
        for path, expected in self.inputs.items():
            if sha256(path) != expected:
                raise ValueError(f"Input changed during audit: {path}")

    def safe_output(self, parent):
        parent = Path(parent).resolve()
        protected = [self.index_file.parent, *self.folders.values()]
        if any(parent.is_relative_to(path) or path.is_relative_to(parent) for path in protected):
            raise ValueError("Choose an output folder separate from the upstream run folders")
        return parent


class PilotResultAudit:
    """Reconcile saved prices and expose descriptive diagnostics, without selection changes."""

    FLAGS = ["preferred_side_missing", "band_disjoint_from_call_bounds", "midpoint_outside_call_bounds"]

    def __init__(self, settings=None):
        self.settings = settings or ResultAuditSettings()

    def checked(self, frames):
        data = {name: frame.copy(deep=True) for name, frame in frames.items()}
        manifest = data["model_manifest"]
        require(manifest, ["quote_date", "root", "status", "input_quotes"],
                "manifest", ["quote_date", "root"])
        # Carry preparation may also contain the next month's exit observations.
        # Scope calibration diagnostics to the indexed model dates and roots.
        for name in ["calibration_quotes", "primary_carry"]:
            data[name] = data[name].merge(manifest[["quote_date", "root"]],
                on=["quote_date", "root"], how="inner", validate="many_to_one")
        q, c, r = (data[name] for name in ("calibration_quotes", "primary_carry", "quote_residuals"))
        p, raw, b = (data[name] for name in ("delta_panel", "pilot_quotes", "selection_buckets"))
        require(q, STRIKE_KEY + self.FLAGS + ["source_kind", "source_bid", "source_ask",
            "call_bid", "call_ask", "call_mid", "call_half_width", "forward", "discount_factor", "case"],
            "calibration quotes", STRIKE_KEY)
        require(r, list(q.columns) + ["model_call_price", "price_residual_points",
            "residual_half_spreads", "outside_original_band"], "quote residuals", STRIKE_KEY)
        require(c, GROUP + ["forward", "discount_factor", "spot", "annual_rate", "maturity_years",
            "parity_bands_incompatible", "fitted_parity_outside_bands", "carry_ready", "case"], "carry", GROUP)
        require(p, SIDE_KEY + ["entry_id", "bid", "ask", "mid", "underlying_last", "calendar_days",
            "ah_price", "ah_status", "black_status", "end_status", "forward", "discount_factor", "case"],
            "delta panel", ["entry_id"])
        require(raw, SIDE_KEY + ["bid", "ask", "mid", "underlying_last"], "observed quotes", SIDE_KEY)
        require(b, ["quote_date", "status"], "selection buckets")
        require(data["expiry_summary"], GROUP + ["optimizer_evaluations", "optimizer_message",
            "active_proxy_bounds", "rms_half_spreads", "outside_original_bands"], "expiry summary", GROUP)
        if not p.kind.isin(["call", "put"]).all() or p.duplicated(SIDE_KEY).any():
            raise ValueError("Invalid or duplicate panel contracts")
        for frame in (q, r):
            for name in self.FLAGS:
                frame[name] = boolean(frame[name])
        r["outside_original_band"] = boolean(r.outside_original_band)
        for name in ["carry_ready", "parity_bands_incompatible", "fitted_parity_outside_bands"]:
            c[name] = boolean(c[name])
        for frame, names in ((q, ["strike", "source_bid", "source_ask", "call_bid", "call_ask", "call_mid", "call_half_width", "forward", "discount_factor"]),
                             (r, ["model_call_price", "price_residual_points", "residual_half_spreads"]),
                             (p, ["strike", "bid", "ask", "mid", "underlying_last", "calendar_days", "forward", "discount_factor"]),
                             (c, ["forward", "discount_factor", "spot", "annual_rate", "maturity_years"])):
            if not np.isfinite(frame[names].to_numpy(float)).all():
                raise ValueError("Nonfinite saved calibration, carry or entry values")
        if (q.call_half_width.le(0).any() or p.ask.le(p.bid).any() or p.bid.le(0).any()
                or not q.source_kind.isin(["call", "put"]).all()
                or c[["forward", "discount_factor", "spot", "maturity_years"]].le(0).any().any()):
            raise ValueError("Invalid saved quote widths or carry")
        # Reconcile shared training targets; never treat an opposite-side quote as the entry quote.
        shared = q.merge(r, on=STRIKE_KEY, suffixes=("", "_residual"), validate="one_to_one")
        fitted_targets = q.merge(manifest.loc[manifest.status.eq("fitted"), ["quote_date", "root"]],
                                on=["quote_date", "root"], validate="many_to_one")
        if len(shared) != len(fitted_targets) or len(shared) != len(r):
            raise ValueError("Calibration quotes and residual identities differ")
        for name in ["source_bid", "source_ask", "call_bid", "call_ask", "call_mid", "call_half_width", "forward", "discount_factor"]:
            close(shared[name], shared[name + "_residual"], f"saved target {name}")
        for name in ["source_kind", "case"] + self.FLAGS:
            if not shared[name].eq(shared[name + "_residual"]).all():
                raise ValueError(f"Saved target metadata differs: {name}")
        offset = np.where(q.source_kind.eq("put"), q.discount_factor * (q.forward - q.strike), 0)
        close(q.call_bid, q.source_bid + offset, "equivalent-call bid")
        close(q.call_ask, q.source_ask + offset, "equivalent-call ask")
        close(q.call_mid, (q.call_bid + q.call_ask) / 2, "equivalent-call midpoint")
        close(q.call_half_width, (q.call_ask - q.call_bid) / 2, "equivalent-call half-spread")
        close(r.price_residual_points, r.model_call_price - r.call_mid, "AH residual")
        close(r.residual_half_spreads, r.price_residual_points / r.call_half_width, "AH normalized residual")
        tol = self.settings.quote_band_tolerance
        outside = (r.model_call_price < r.call_bid-tol) | (r.model_call_price > r.call_ask+tol)
        if not outside.eq(r.outside_original_band).all():
            raise ValueError("Saved outside-band flags disagree with the requested tolerance")
        for frame, label in ((q, "calibration"), (p, "panel")):
            merged = frame.merge(c[GROUP + ["forward", "discount_factor", "case"]],
                                 on=GROUP, how="left", suffixes=("", "_carry"), validate="many_to_one")
            close(merged.forward, merged.forward_carry, f"{label} forward")
            close(merged.discount_factor, merged.discount_factor_carry, f"{label} discount")
            if not merged.case.eq(merged.case_carry).all():
                raise ValueError(f"{label} carry scenario differs")
        source = p.merge(raw[SIDE_KEY + ["bid", "ask", "mid", "underlying_last"]],
                         on=SIDE_KEY, how="left", suffixes=("", "_observed"), validate="one_to_one")
        for name in ["bid", "ask", "mid", "underlying_last"]:
            close(source[name], source[name + "_observed"], f"entry observed {name}")
        close(p.mid, (p.bid+p.ask)/2, "entry midpoint")
        ready = p.ah_status.eq("ready")
        if not np.isfinite(p.loc[ready, "ah_price"].to_numpy(float)).all():
            raise ValueError("Nonfinite price for a ready AH entry")
        return data

    def parity_pairs(self, raw, carry, window):
        raw = raw.loc[raw.quote_date.isin(carry.quote_date)]
        cols = STRIKE_KEY + ["bid", "ask", "mid", "underlying_last"]
        calls = raw.loc[raw.kind.eq("call"), cols]
        puts = raw.loc[raw.kind.eq("put"), cols]
        pairs = calls.merge(puts, on=STRIKE_KEY, suffixes=("_call", "_put"), validate="one_to_one")
        pairs = pairs.merge(carry[GROUP + ["forward", "discount_factor", "spot"]],
                            on=GROUP, how="inner", validate="many_to_one")
        names = ["bid_call", "ask_call", "mid_call", "bid_put", "ask_put", "mid_put"]
        if not np.isfinite(pairs[names].to_numpy(float)).all():
            raise ValueError("Nonfinite matched observed parity quotes")
        if ((pairs.ask_call <= pairs.bid_call) | (pairs.ask_put <= pairs.bid_put)).any():
            raise ValueError("Invalid matched observed parity spreads")
        close(pairs.underlying_last_call, pairs.spot, "parity-pair call spot")
        close(pairs.underlying_last_put, pairs.spot, "parity-pair put spot")
        pairs["spot_log_moneyness"] = np.log(pairs.strike/pairs.spot)
        pairs["within_parity_window"] = pairs.spot_log_moneyness.abs() <= window + 1e-12
        pairs["parity_bid"] = pairs.bid_call-pairs.ask_put
        pairs["parity_ask"] = pairs.ask_call-pairs.bid_put
        pairs["assumed_parity_value"] = pairs.discount_factor*(pairs.forward-pairs.strike)
        pairs["observed_mid_parity_residual"] = pairs.mid_call-pairs.mid_put-pairs.assumed_parity_value
        pairs["parity_half_width"] = (pairs.parity_ask-pairs.parity_bid)/2
        pairs["parity_residual_half_widths"] = -pairs.observed_mid_parity_residual/pairs.parity_half_width
        tol = self.settings.quote_band_tolerance
        pairs["outside_parity_band"] = ((pairs.assumed_parity_value < pairs.parity_bid-tol)
                                        | (pairs.assumed_parity_value > pairs.parity_ask+tol))
        pairs["forward_band_lower"] = pairs.strike+pairs.parity_bid/pairs.discount_factor
        pairs["forward_band_upper"] = pairs.strike+pairs.parity_ask/pairs.discount_factor
        return pairs

    def selected_contracts(self, panel, residuals, carry):
        target_cols = STRIKE_KEY + ["source_kind", "model_call_price", "residual_half_spreads",
            "outside_original_band", "call_bid", "call_ask", "call_half_width"] + self.FLAGS
        targets = residuals[target_cols].rename(columns={name: "training_" + name
            for name in target_cols if name not in STRIKE_KEY})
        entries = panel.merge(targets, on=STRIKE_KEY, how="left", validate="many_to_one", indicator=True)
        entries["original_price_status"] = np.where(entries._merge.eq("both"),
                                                       "saved_calibration_strike", "no_saved_calibration_strike")
        entries = entries.drop(columns="_merge")
        entries["entry_half_spread"] = (entries.ask-entries.bid)/2
        parity = entries.discount_factor*(entries.forward-entries.strike)
        entries["original_ah_price"] = entries.training_model_call_price-np.where(entries.kind.eq("put"), parity, 0)
        entries["entry_side_is_training_side"] = entries.kind.eq(entries.training_source_kind).astype("boolean")
        entries.loc[entries.original_price_status.ne("saved_calibration_strike"), "entry_side_is_training_side"] = pd.NA
        entries["smoothed_price_status"] = np.where(entries.ah_status.eq("ready"), "saved_ready_price", "AH_entry_not_ready")
        prices = {"original": entries.original_ah_price,
                  "smoothed": entries.ah_price.where(entries.ah_status.eq("ready"))}
        for label, values in prices.items():
            entries[f"{label}_price_minus_entry_mid"] = values-entries.mid
            entries[f"{label}_residual_entry_half_spreads"] = (values-entries.mid)/entries.entry_half_spread
            available = values.notna()
            outside = ((values < entries.bid-self.settings.quote_band_tolerance)
                       | (values > entries.ask+self.settings.quote_band_tolerance)).astype("boolean")
            entries[f"{label}_outside_entry_band"] = outside.where(available, pd.NA)
        entries["smoothed_minus_original_price"] = prices["smoothed"]-prices["original"]
        fields = GROUP + ["parity_bands_incompatible", "fitted_parity_outside_bands"]
        entries = entries.merge(carry[fields].rename(columns={name: "expiry_"+name
            for name in fields if name not in GROUP}), on=GROUP, how="left", validate="many_to_one")
        entries["focus_date"] = entries.quote_date.isin(self.settings.focus_dates)
        return entries

    def selection(self, buckets, panel, settings, manifest):
        b = buckets.copy()
        counts = {"target_days": len(settings["target_days"]),
                  "target_spot_y": len(settings["target_spot_y"]), "kind": len(settings["kinds"])}
        expected = int(np.prod(list(counts.values())))
        resolved = b.status.isin(["selected", "duplicate_bucket"])
        if not set(b.status).issubset({"selected", "duplicate_bucket", "side_unavailable",
                                      "no_current_entries", "no_current_candidates"}):
            raise ValueError("Unknown saved selection status")
        if "entry_id" not in b:
            b["entry_id"] = np.nan
        require(b.loc[resolved], ["root", "entry_id", "target_days", "target_spot_y", "kind", "strike", "expire_date"],
                "resolved selection buckets")
        joined = b.loc[resolved].merge(panel[["entry_id"] + SIDE_KEY + ["calendar_days"]],
            on="entry_id", how="left", suffixes=("", "_entry"), validate="many_to_one", indicator=True)
        if joined._merge.ne("both").any():
            raise ValueError("Resolved selection bucket points to an absent entry")
        for name in GROUP + ["kind"]:
            if not joined[name].eq(joined[name + "_entry"]).all():
                raise ValueError(f"Bucket/entry mismatch: {name}")
        close(joined.strike, joined.strike_entry, "bucket entry strike")
        first = b.loc[b.status.eq("selected"), "entry_id"]
        if first.duplicated().any() or set(first) != set(panel.entry_id):
            raise ValueError("Selected bucket identities differ from the saved panel")
        b = b.merge(panel[["entry_id", "calendar_days"]], on="entry_id", how="left", validate="many_to_one")
        if "target_days" not in b:
            b["target_days"] = np.nan
        b["days_minus_target"] = b.calendar_days-pd.to_numeric(b.target_days)
        b["selection_focus_date"] = b.quote_date.isin(self.settings.selection_dates)
        dates = sorted(set(manifest.quote_date) | set(b.quote_date) | set(panel.quote_date))
        daily = []
        for date in dates:
            part, entries = b.loc[b.quote_date.eq(date)], panel.loc[panel.quote_date.eq(date)]
            status_counts = part.status.value_counts()
            roots = int(part.root.nunique()) if "root" in part else 0
            daily.append({"quote_date": date, "expected_buckets_per_eligible_root": expected,
                "roots_with_bucket_records": roots, "recorded_buckets": len(part),
                "selected_buckets": int(status_counts.get("selected", 0)),
                "duplicate_buckets": int(status_counts.get("duplicate_bucket", 0)),
                "side_unavailable_buckets": int(status_counts.get("side_unavailable", 0)),
                "no_current_entries_records": int(status_counts.get("no_current_entries", 0)),
                "no_current_candidates_records": int(status_counts.get("no_current_candidates", 0)),
                "unique_entries": len(entries), "unique_expiries": entries.expire_date.nunique(),
                "both_deltas_ready": int((entries.ah_status.eq("ready") & entries.black_status.eq("ready")).sum()),
                "matched_endpoints": int(entries.end_status.eq("matched").sum()),
                "selection_focus_date": date in self.settings.selection_dates})
        target = b.loc[b.target_days.notna()].groupby(["quote_date", "target_days", "status"],
            dropna=False).size().rename("buckets").reset_index()
        return b, pd.DataFrame(daily), target

    @staticmethod
    def metrics(frame):
        residual = frame.residual_half_spreads.to_numpy(float)
        return {"calibration_quotes": len(frame), "rms_half_spreads": float(np.sqrt(np.mean(residual**2))),
                "max_half_spreads": float(np.abs(residual).max()),
                "outside_original_bands": int(frame.outside_original_band.sum()),
                "residual_squares_sum": float(np.sum(residual**2)),
                **{name: int(frame[name].sum()) for name in PilotResultAudit.FLAGS}}

    def calibration_tables(self, residuals, carry, expiry_summary, manifest, parity):
        expiry = []
        total_sq = float((residuals.residual_half_spreads**2).sum())
        for key, part in residuals.groupby(GROUP, sort=True):
            expiry.append({**dict(zip(GROUP, key)), **self.metrics(part),
                "contribution_to_pooled_calibration_mse": float((part.residual_half_spreads**2).sum()/len(residuals)),
                "share_of_residual_squares": float((part.residual_half_spreads**2).sum()/total_sq) if total_sq else 0})
        expiry = pd.DataFrame(expiry)
        fields = GROUP + ["forward", "discount_factor", "spot", "annual_rate", "maturity_years",
            "parity_bands_incompatible", "fitted_parity_outside_bands", "carry_ready"]
        optional = [name for name in ["pairs", "outside_parity_bands", "rms_parity_half_widths",
            "forward_band_lower", "forward_band_upper", "minimum_band_multiplier"] if name in carry]
        expiry = expiry.merge(carry[fields + optional], on=GROUP, validate="one_to_one")
        optimizer = GROUP + ["optimizer_evaluations", "optimizer_message", "active_proxy_bounds"]
        expiry = expiry.merge(expiry_summary[optimizer], on=GROUP, validate="one_to_one")
        expiry["days"] = expiry.maturity_years*365
        expiry["focus_date"] = expiry.quote_date.isin(self.settings.focus_dates)
        pair_summaries = []
        for key, part in parity.groupby(GROUP, sort=True):
            window = part.loc[part.within_parity_window]
            pair_summaries.append({**dict(zip(GROUP, key)), "matched_observed_pairs": len(part),
                "observed_pairs_outside_parity_bands": int(part.outside_parity_band.sum()),
                "window_pairs": len(window), "window_pairs_outside_parity_bands": int(window.outside_parity_band.sum()),
                "window_forward_intersection_lower": window.forward_band_lower.max(),
                "window_forward_intersection_upper": window.forward_band_upper.min()})
        pairs = pd.DataFrame(pair_summaries, columns=GROUP + ["matched_observed_pairs",
            "observed_pairs_outside_parity_bands", "window_pairs", "window_pairs_outside_parity_bands",
            "window_forward_intersection_lower", "window_forward_intersection_upper"])
        expiry = expiry.merge(pairs, on=GROUP, how="left", validate="one_to_one")
        daily = []
        for row in manifest.to_dict("records"):
            part = residuals.loc[residuals.quote_date.eq(row["quote_date"]) & residuals.root.eq(row["root"])]
            groups = carry.loc[carry.quote_date.eq(row["quote_date"]) & carry.root.eq(row["root"])]
            metrics = self.metrics(part) if len(part) else dict(calibration_quotes=0, rms_half_spreads=np.nan,
                max_half_spreads=np.nan, outside_original_bands=0, residual_squares_sum=0,
                **{name: 0 for name in self.FLAGS})
            if row["status"] == "fitted":
                if len(part) != int(row["input_quotes"]):
                    raise ValueError("Manifest input quote count differs from saved residuals")
                for name in ["rms_half_spreads", "outside_original_bands"]:
                    if name in row:
                        close(metrics[name], row[name], f"manifest {name}")
            elif len(part):
                raise ValueError("Residuals exist for a non-fitted manifest row")
            daily.append({"quote_date": row["quote_date"], "root": row["root"],
                "model_status": row["status"], **metrics,
                "expiry_groups": len(groups), "parity_incompatible_groups": int(groups.parity_bands_incompatible.sum()),
                "fitted_parity_outside_groups": int(groups.fitted_parity_outside_bands.sum()),
                "focus_date": row["quote_date"] in self.settings.focus_dates})
        return expiry, pd.DataFrame(daily)

    def attribution(self, entries, paired, saved, coverage):
        keys = ["scenario", "entry_id"]
        require(paired, keys + SIDE_KEY + ["ah_net_pnl", "black_net_pnl",
            "abs_error_improvement", "squared_error_improvement"], "paired results", keys)
        require(saved, keys + ["ah_net_pnl", "black_net_pnl", "delta_difference", "spot_change",
            "pnl_gap", "delta_exposure", "funding_difference", "cost_effect", "identity_residual",
            "abs_error_improvement", "squared_error_improvement", "maturity_band"], "saved attribution", keys)
        require(coverage, keys + ["status"], "comparison coverage", keys)
        compared = coverage.loc[coverage.status.eq("compared"), keys]
        for source in (paired, saved):
            if set(map(tuple, source[keys].to_numpy())) != set(map(tuple, compared.to_numpy())):
                raise ValueError("Compared coverage identities differ from saved P&L or attribution")
        if not set(paired.entry_id).issubset(entries.entry_id):
            raise ValueError("Compared P&L refers to an absent entry")
        metadata = paired.merge(entries[["entry_id"] + SIDE_KEY], on="entry_id", suffixes=("", "_entry"), validate="many_to_one")
        for name in GROUP + ["kind"]:
            if not metadata[name].eq(metadata[name + "_entry"]).all():
                raise ValueError(f"Comparison/entry metadata mismatch: {name}")
        close(metadata.strike, metadata.strike_entry, "comparison entry strike")
        fields = ["ah_net_pnl", "black_net_pnl", "abs_error_improvement", "squared_error_improvement"]
        joined = paired[keys + fields].merge(saved[keys + fields], on=keys,
            suffixes=("", "_saved"), validate="one_to_one")
        for name in fields:
            close(joined[name], joined[name + "_saved"], f"attribution {name}")
        close(paired.abs_error_improvement, paired.black_net_pnl.abs()-paired.ah_net_pnl.abs(), "paired MAE")
        close(paired.squared_error_improvement, paired.black_net_pnl**2-paired.ah_net_pnl**2, "paired MSE")
        close(saved.delta_exposure, saved.delta_difference*saved.spot_change, "delta exposure")
        close(saved.pnl_gap, saved.ah_net_pnl-saved.black_net_pnl, "P&L gap")
        close(saved.pnl_gap, saved.delta_exposure+saved.funding_difference+saved.cost_effect, "P&L attribution")
        close(saved.identity_residual, 0, "saved identity residual")
        mid = saved.loc[saved.scenario.eq("mid")]
        fields = ["entry_id", "ah_net_pnl", "black_net_pnl", "abs_error_improvement", "squared_error_improvement",
            "delta_difference", "spot_change", "delta_exposure", "funding_difference", "maturity_band"]
        out = entries.merge(mid[fields], on="entry_id", how="left", validate="one_to_one", indicator=True)
        out["midpoint_comparison_status"] = np.where(out._merge.eq("both"), "compared", "not_compared")
        return out.drop(columns="_merge")

    def run(self, frames, panel_settings, parity_window=0.03):
        data = self.checked(frames)
        r, c, panel = (data[name] for name in ["quote_residuals", "primary_carry", "delta_panel"])
        if r.empty or not 0 < parity_window < 1:
            raise ValueError("Require saved fitted residuals and a valid parity window")
        pairs = self.parity_pairs(data["pilot_quotes"], c, parity_window)
        entries = self.selected_contracts(panel, r, c)
        buckets, selection, targets = self.selection(data["selection_buckets"], panel,
                                                     panel_settings, data["model_manifest"])
        expiry, daily = self.calibration_tables(r, c, data["expiry_summary"], data["model_manifest"], pairs)
        mid = self.attribution(entries, data["paired_results"], data["attribution"], data["coverage"])
        residuals = r.copy()
        counts = panel.groupby(STRIKE_KEY).size().rename("selected_entry_count").reset_index()
        residuals = residuals.merge(counts, on=STRIKE_KEY, how="left", validate="one_to_one")
        residuals["selected_entry_count"] = residuals.selected_entry_count.fillna(0).astype(int)
        residuals["focus_date"] = residuals.quote_date.isin(self.settings.focus_dates)
        price_metrics = []
        for (date, root), part in entries.groupby(["quote_date", "root"], sort=True):
            row = {"quote_date": date, "root": root, "selected_entries": len(part),
                   "original_price_missing": int(part.original_ah_price.isna().sum()),
                   "smoothed_price_not_ready": int(part.smoothed_price_status.ne("saved_ready_price").sum())}
            for label in ["original", "smoothed"]:
                values = part[f"{label}_residual_entry_half_spreads"].dropna()
                row[f"{label}_priced_entries"] = len(values)
                row[f"{label}_entry_rms_half_spreads"] = float(np.sqrt(np.mean(values**2))) if len(values) else np.nan
                row[f"{label}_outside_entry_bands"] = int(part[f"{label}_outside_entry_band"].sum())
            price_metrics.append(row)
        daily = daily.merge(pd.DataFrame(price_metrics), on=["quote_date", "root"], how="left", validate="one_to_one")
        contributions = data["daily_attribution"].loc[lambda frame: frame.scenario.eq("mid")]
        require(contributions, ["quote_date", "comparisons", "mae_improvement", "mse_improvement",
            "pooled_mae_contribution", "pooled_mse_contribution"], "daily attribution", ["quote_date"])
        # This date-level performance diagnostic is repeated across roots, never pooled a second time.
        daily = daily.merge(contributions[["quote_date", "comparisons", "mae_improvement", "mse_improvement",
            "pooled_mae_contribution", "pooled_mse_contribution"]], on="quote_date", how="left", validate="many_to_one")
        focus_dates = sorted(set(self.settings.focus_dates) | set(self.settings.selection_dates))
        focus_status = pd.DataFrame({"quote_date": focus_dates,
            "calibration_focus": [date in self.settings.focus_dates for date in focus_dates],
            "selection_focus": [date in self.settings.selection_dates for date in focus_dates],
            "present_in_manifest": [date in set(data["model_manifest"].quote_date) for date in focus_dates]})
        tables = {"daily_audit": daily, "expiry_audit": expiry, "quote_audit": residuals,
            "parity_pairs": pairs, "selected_contract_audit": entries, "selection_buckets": buckets,
            "selection_daily": selection, "selection_target_summary": targets,
            "midpoint_entry_audit": mid, "long_maturity_entries": mid.loc[mid.calendar_days.between(40, 45)].copy(),
            "saved_daily_attribution": data["daily_attribution"], "saved_group_summary": data["group_summary"],
            "comparison_coverage": data["coverage"], "focus_status": focus_status}
        for name, frame in data.items():
            if name.startswith("validation_"):
                tables[name] = frame
        summary = {"calibration_quotes": len(r), "original_outside_bands": int(r.outside_original_band.sum()),
            "focus_outside_bands": int(r.loc[r.quote_date.isin(self.settings.focus_dates), "outside_original_band"].sum()),
            "selected_entries": len(entries), "midpoint_compared": int(mid.midpoint_comparison_status.eq("compared").sum()),
            "original_price_missing": int(entries.original_ah_price.isna().sum()),
            "all_entries_retained": len(entries) == len(panel),
            "pooled_original_rms_half_spreads": float(np.sqrt(np.mean(r.residual_half_spreads**2)))}
        return tables, summary


class ResultAuditReport:
    """Static scientific figures and a notebook that reads only this saved audit."""

    def __init__(self, tables, settings):
        self.tables, self.settings = tables, settings

    def plots(self, folder):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        folder.mkdir()
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
        daily = self.tables["daily_audit"]
        dates = sorted(daily.quote_date.unique())
        x = np.arange(len(dates))
        labels = [date[5:] for date in dates]

        def save(fig, name):
            fig.tight_layout()
            fig.savefig(folder / name, dpi=160, bbox_inches="tight")
            plt.close(fig)

        # Quote counts and squared-residual sums are pooled, never daily RMS averages.
        pooled = daily.groupby("quote_date").agg(n=("calibration_quotes", "sum"),
            squares=("residual_squares_sum", "sum"), outside=("outside_original_bands", "sum")).reindex(dates)
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        colors = ["#bc5a36" if date in self.settings.focus_dates else "#2373a1" for date in dates]
        axes[0].bar(x, np.sqrt(pooled.squares/pooled.n), color=colors)
        axes[0].set(ylabel="RMS half-spreads", title="Original AH calibration and saved midpoint hedge diagnostics")
        axes[1].bar(x, pooled.outside, color=colors)
        axes[1].set_ylabel("Outside-band quotes")
        mid = self.tables["saved_daily_attribution"].query("scenario == 'mid'").set_index("quote_date").reindex(dates)
        axes[2].bar(x, mid.pooled_mse_contribution, color=np.where(mid.pooled_mse_contribution >= 0, "#2373a1", "#bc5a36"))
        axes[2].axhline(0, color="black", lw=0.8)
        axes[2].set(ylabel="Pooled MSE contribution", xlabel="Entry date; positive contribution favours AH")
        axes[2].set_xticks(x, labels, rotation=60)
        fig.suptitle("Co-occurrence is descriptive; this figure does not establish causation", fontsize=10)
        save(fig, "01_calibration_and_hedge_dates.png")

        q = self.tables["quote_audit"]
        focus = [date for date in self.settings.focus_dates if date in set(q.quote_date)]
        if not focus:
            focus = dates[:1]
        fig, axes = plt.subplots(len(focus), 1, figsize=(10, 3.3*len(focus)), squeeze=False)
        for ax, date in zip(axes.flat, focus):
            part = q.loc[q.quote_date.eq(date)]
            for kind, color in [("call", "#2373a1"), ("put", "#bc5a36")]:
                side = part.loc[part.source_kind.eq(kind)]
                ax.scatter(np.log(side.strike/side.forward), side.residual_half_spreads,
                           s=10, alpha=0.5, color=color, label=f"{kind.title()} source quotes")
            selected = part.loc[part.selected_entry_count.gt(0)]
            ax.scatter(np.log(selected.strike/selected.forward), selected.residual_half_spreads,
                       s=55, facecolors="none", edgecolors="black", label="Selected hedge strikes")
            for bound in (-1, 1):
                ax.axhline(bound, color="gray", lw=0.8, ls="--")
            ax.set(title=f"{date}: all fitted expiries; original AH targets", ylabel="Residual / original half-spread",
                   xlabel="log(K / same-expiry forward)")
            ax.legend(fontsize=8, ncol=3)
        save(fig, "02_focus_calibration_residuals.png")

        entries = self.tables["selected_contract_audit"]
        part = entries.loc[entries.quote_date.isin(focus)].sort_values(SIDE_KEY).reset_index(drop=True)
        if part.empty:
            part = entries.sort_values(SIDE_KEY).head(18).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        xx = np.arange(len(part))
        ax.scatter(xx, part.original_residual_entry_half_spreads, s=22, color="#2373a1", label="Original AH, own entry quote")
        ax.scatter(xx, part.smoothed_residual_entry_half_spreads, s=22, marker="x", color="#bc5a36", label="Saved smoothed price, own entry quote")
        for bound in (-1, 1):
            ax.axhline(bound, color="gray", lw=0.8, ls="--")
        ax.set(title="Selected contracts: price fit, not realised hedge error", ylabel="Residual / own entry half-spread",
               xlabel="Selected contract index; identities and both option kinds remain in the CSV")
        ax.legend()
        save(fig, "03_selected_contract_price_fit.png")

        selection = self.tables["selection_target_summary"]
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for status, color in [("selected", "#2373a1"), ("duplicate_bucket", "#bc5a36"), ("side_unavailable", "#777777")]:
            values = selection.loc[selection.status.eq(status)].groupby("quote_date").buckets.sum().reindex(dates, fill_value=0)
            offset = {"selected": -0.25, "duplicate_bucket": 0, "side_unavailable": 0.25}[status]
            ax.bar(x+offset, values, width=0.25, color=color, label=status.replace("_", " "))
        ax.set_xticks(x, labels, rotation=60)
        ax.set(title="Saved basket selection: unique contracts, repeated targets and unavailable sides",
               ylabel="Target buckets", xlabel="Entry date")
        ax.legend()
        save(fig, "04_selection_coverage.png")

    def notebook(self, path):
        cells = []

        def markdown(text):
            cells.append(dict(cell_type="markdown", metadata={}, source=text.splitlines(True)))

        def code(text):
            cells.append(dict(cell_type="code", metadata={}, execution_count=None, outputs=[], source=text.splitlines(True)))

        markdown("# Saved-results calibration and hedge audit\n\n"
                 "This notebook reads saved audit tables. It does not calibrate models, solve a PDE, "
                 "change deltas or rerun the hedge ledger. All entries remain in the original comparison. "
                 "Focus dates are retrospective diagnostics, not new entry rules.\n")
        code("from pathlib import Path\nimport json\nimport pandas as pd\n"
             "from IPython.display import display, Image, Markdown\n\n"
             f"AUDIT_RUN = Path({str(path.parent.resolve())!r})\n"
             "if not (AUDIT_RUN / 'audit.json').exists() and (Path.cwd() / 'audit.json').exists():\n"
             "    AUDIT_RUN = Path.cwd()\n"
             "audit = json.loads((AUDIT_RUN / 'audit.json').read_text())\n"
             "def table(name):\n"
             "    try:\n        return pd.read_csv(AUDIT_RUN / f'{name}.csv')\n"
             "    except pd.errors.EmptyDataError:\n        return pd.DataFrame()\n"
             "pd.set_option('display.max_columns', 40)\n"
             "display(pd.DataFrame([audit['summary']]))\n"
             "display(table('focus_status'))\n")
        markdown("## Daily calibration and carry\n\n"
                 "RMS residuals use original equivalent-call half-spreads. Outside-band counts use the saved "
                 "price tolerance. A successful model fit is not an endorsement of quote fit or Greeks.\n")
        code("daily = table('daily_audit')\ndisplay(daily)\n"
             "display(Image(filename=str(AUDIT_RUN / 'plots/01_calibration_and_hedge_dates.png')))\n")
        markdown("## Inspect a focus date\n\n"
                 "Edit `FOCUS_DATE` to inspect any available model date. The full-date tables remain saved. "
                 "Forward-band intersections below describe same-date observed parity pairs at the assumed discount. "
                 "The original producer carry flags remain separate.\n")
        focus = self.tables["daily_audit"].loc[lambda f: f.focus_date, "quote_date"]
        selected_date = focus.iloc[0] if len(focus) else self.tables["daily_audit"].quote_date.iloc[0]
        code(f"FOCUS_DATE = {selected_date!r}\n"
             "expiry = table('expiry_audit')\n"
             "display(expiry.loc[expiry.quote_date.eq(FOCUS_DATE)].sort_values('contribution_to_pooled_calibration_mse', ascending=False))\n"
             "pairs = table('parity_pairs')\n"
             "display(pairs.loc[pairs.quote_date.eq(FOCUS_DATE) & pairs.within_parity_window].sort_values('parity_residual_half_widths', key=abs, ascending=False).head(30))\n"
             "display(Image(filename=str(AUDIT_RUN / 'plots/02_focus_calibration_residuals.png')))\n")
        markdown("## Prices for the actual selected contracts\n\n"
                 "Both original AH and saved smoothed prices are compared with each entry's own call or put bid/ask. "
                 "Original put prices are derived from the saved call price using the same-date parity term. "
                 "Missing original strike prices and non-ready smoothed prices remain explicit. "
                 "Smoothed minus original includes the diffusion change and numerical error; it is not pure PDE error.\n")
        code("contracts = table('selected_contract_audit')\n"
             "columns = ['quote_date','expire_date','strike','kind','training_source_kind','entry_side_is_training_side',"
             "'bid','ask','original_ah_price','ah_price','original_residual_entry_half_spreads',"
             "'smoothed_residual_entry_half_spreads','original_outside_entry_band','smoothed_outside_entry_band',"
             "'smoothed_minus_original_price','expiry_parity_bands_incompatible','original_price_status','smoothed_price_status']\n"
             "display(contracts.loc[contracts.quote_date.eq(FOCUS_DATE), columns])\n"
             "display(Image(filename=str(AUDIT_RUN / 'plots/03_selected_contract_price_fit.png')))\n")
        markdown("## Basket coverage\n\n"
                 "A duplicate bucket means two target requests mapped to the same fixed contract. "
                 "An unavailable side is a different status. Neither is inferred from realised hedge results.\n")
        code("display(table('selection_daily'))\n"
             "buckets = table('selection_buckets')\n"
             "display(buckets.loc[buckets.selection_focus_date])\n"
             "display(Image(filename=str(AUDIT_RUN / 'plots/04_selection_coverage.png')))\n")
        markdown("## Saved hedge attribution and longer maturities\n\n"
                 "These are descriptive results on the complete original sample. Lower absolute P&L and "
                 "zero-centred RMS are the hedge-error measures. Higher signed profit is not a hedge-quality objective. "
                 "No dates or maturity groups are removed, and no IID-row inference is performed.\n")
        code("display(table('saved_daily_attribution'))\n"
             "display(table('saved_group_summary'))\n"
             "longer = table('long_maturity_entries')\n"
             "cols = ['quote_date','expire_date','strike','kind','calendar_days','original_residual_entry_half_spreads',"
             "'smoothed_residual_entry_half_spreads','ah_net_pnl','black_net_pnl','abs_error_improvement',"
             "'squared_error_improvement','midpoint_comparison_status']\n"
             "display(longer.reindex(columns=cols))\n")
        markdown("## Numerical validation coverage\n\n"
                 "The tables below reuse existing validation runs. Generated representative contracts are "
                 "not independent validation of every selected entry's Greeks. Missing or unrequested validation "
                 "dates remain visible. Full-domain shape flags and report-window flags retain their original scope.\n")
        code("for name in ['validation_coverage','validation_quote_fit','validation_sensitivity','validation_forward_shapes','validation_forward_backward']:\n"
             "    display(Markdown(f'**{name}**'))\n    display(table(name))\n")
        markdown("## Evidence limits\n\n"
                 "Contract identity and snapshot provenance remain unverified. Pricing carry and funding are assumed; "
                 "the hedge is a synthetic fractional index position. The report neither certifies economic wing "
                 "robustness nor claims executable returns or out-of-sample hedge superiority.\n")
        document = dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}}, nbformat=4, nbformat_minor=5)
        # Cell ids keep notebook validators quiet across supported Jupyter versions.
        for i, cell in enumerate(cells):
            cell["id"] = f"saved-audit-{i:02d}"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
