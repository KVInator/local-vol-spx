"""Selected-contract numerical experiments; never change a frozen study."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict, replace
from pathlib import Path

from artifacts import atomic_json, sha256
import numpy as np
import pandas as pd

from pde import AHBackwardPricer
from pricing import BlackPricer, invert_time_value
from surface import AHGrid, AHShortEndVariance, AndreasenHugeCalibrator


def contract_panel(model, carry, log_strikes):
    """First, nearest 35-day and last pillars, with both payoff kinds."""
    carry = carry.sort_values("maturity_years").reset_index(drop=True)
    if len(carry) != len(model.maturities) or not np.allclose(
        carry.maturity_years, model.maturities, rtol=0, atol=1e-12
    ):
        raise ValueError("Carry must match the saved model's maturity pillars.")
    if not np.allclose(carry.forward, model.forwards, rtol=1e-12) or not np.allclose(
        carry.discount_factor, model.discounts, rtol=1e-12
    ):
        raise ValueError("Carry must match the saved model's forwards and discounts.")
    indices = sorted(
        {0, int(np.argmin(abs(365 * model.maturities - 35))), len(model.maturities) - 1}
    )
    rows = []
    for i in indices:
        for y in log_strikes:
            for kind in ("call", "put"):
                rows.append(
                    dict(
                        expire_date=carry.expire_date.iloc[i],
                        maturity=float(model.maturities[i]),
                        log_strike=float(y),
                        strike=float(model.forwards[i] * np.exp(y)),
                        kind=kind,
                    )
                )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class GreekPanelSettings:
    radii: tuple = (0.0, 0.00025, 0.0005, 0.001)
    log_strikes: tuple = (-0.06, -0.02, 0.0, 0.02, 0.06)
    spot_changes: tuple = (-0.01, 0.0, 0.01)
    bump_fractions: tuple = (0.001, 0.0005, 0.00025, 0.000125)
    width: float = 0.75
    wide_width: float = 0.9
    coarse_intervals: int = 12000
    fine_intervals: int = 24000
    steps_per_day: int = 64

    def __post_init__(self):
        arrays = (self.radii, self.log_strikes, self.spot_changes, self.bump_fractions)
        if any(not len(a) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Panel coordinates must be nonempty and finite.")
        if any(len(set(a)) != len(a) for a in arrays):
            raise ValueError("Panel coordinates must be unique.")
        if (
            min(self.radii) < 0
            or min(self.bump_fractions) <= 0
            or max(self.bump_fractions) >= 1
        ):
            raise ValueError(
                "Use nonnegative radii and spot bumps between zero and one."
            )
        if (
            not np.isfinite([self.width, self.wide_width]).all()
            or not 0.1 <= self.width < self.wide_width
            or not isinstance(self.steps_per_day, int)
            or self.steps_per_day < 1
            or any(
                not isinstance(n, int) or n < 100 or n % 2
                for n in (self.coarse_intervals, self.fine_intervals)
            )
            or self.fine_intervals != 2 * self.coarse_intervals
        ):
            raise ValueError("Use positive settings and two-to-one space refinement.")
        wide = 2 * self.fine_intervals * self.wide_width / self.width
        if abs(wide - round(wide)) > 1e-8 or round(wide) % 2:
            raise ValueError("The wide domain must preserve fine-grid spacing.")

    def cases(self):
        n, steps, width = self.fine_intervals, self.steps_per_day, self.width
        return (
            ("space_coarse", width, self.coarse_intervals, steps, 0.0),
            ("space_fine", width, n, steps, 0.0),
            ("time_fine", width, n, 2 * steps, 0.0),
            ("space_finer", width, 2 * n, 2 * steps, 0.0),
            ("half_cell_shift", width, 2 * n, 2 * steps, width / (2 * n)),
            (
                "domain_wide",
                self.wide_width,
                int(round(2 * n * self.wide_width / width)),
                2 * steps,
                0.0,
            ),
        )


class GreekPanelValidator:
    """Grid, time, domain, spot-bump and short-end blend sensitivity."""

    def __init__(self, settings=None):
        self.settings = settings or GreekPanelSettings()

    def run(self, model, carry, progress=print):
        s = self.settings
        panel = contract_panel(model, carry, s.log_strikes)
        observations, bumps, failures = [], [], []
        for radius in s.radii:
            try:
                variance = AHShortEndVariance(model, radius)
                if s.wide_width + 4 * radius >= variance.max_log_moneyness:
                    raise ValueError(
                        "The blend kernel must remain inside the native domain."
                    )
            except (ValueError, ArithmeticError) as error:
                failures.append(
                    dict(radius=radius, case="coefficient", message=str(error))
                )
                continue
            for case, width, n, steps, shift in s.cases():
                solver = AHBackwardPricer(
                    variance,
                    domain_width=width,
                    space_intervals=n,
                    steps_per_day=steps,
                    early_time_power=3,
                    state_shift=shift,
                )
                for (time, kind), group in panel.groupby(
                    ["maturity", "kind"], sort=True
                ):
                    progress(f"radius={radius:g} {case}: {365*time:.4g}d {kind}...")
                    try:
                        grids = solver.solve_many(
                            group.strike.to_numpy(), time, kind=kind
                        )
                        for contract, grid in zip(group.to_dict("records"), grids):
                            for change in s.spot_changes:
                                spot = model.spot * np.exp(change)
                                values = grid.greeks(spot)
                                row = dict(
                                    contract,
                                    radius=radius,
                                    case=case,
                                    spot_log_change=change,
                                    spot=spot,
                                    price=float(values.price),
                                    delta=float(values.delta),
                                    gamma=float(values.gamma),
                                )
                                if not np.isfinite(
                                    [row[f] for f in ("price", "delta", "gamma")]
                                ).all():
                                    raise ArithmeticError("Nonfinite price or Greek.")
                                scale = float(
                                    model.discount_factor(time)
                                    * model.forward(time)
                                    / model.spot
                                )
                                sign = 1 if kind == "call" else -1
                                lower = max(
                                    sign
                                    * (
                                        scale * spot
                                        - model.discount_factor(time)
                                        * contract["strike"]
                                    ),
                                    0,
                                )
                                upper = (
                                    scale * spot
                                    if kind == "call"
                                    else model.discount_factor(time)
                                    * contract["strike"]
                                )
                                row.update(
                                    negative_gamma=row["gamma"] < -1e-8,
                                    delta_bound_violation=(
                                        row["delta"]
                                        < (-1e-6 if kind == "call" else -scale - 1e-6)
                                        or row["delta"]
                                        > (scale + 1e-6 if kind == "call" else 1e-6)
                                    ),
                                    price_bound_violation=row["price"] < lower - 1e-6
                                    or row["price"] > upper + 1e-6,
                                )
                                observations.append(row)
                                if case == "space_finer":
                                    for fraction in s.bump_fractions:
                                        h = fraction * spot
                                        lo, mid, hi = grid.price(
                                            [spot - h, spot, spot + h]
                                        )
                                        bumps.append(
                                            dict(
                                                row,
                                                bump_points=h,
                                                fd_delta=float((hi - lo) / (2 * h)),
                                                fd_gamma=float(
                                                    (hi - 2 * mid + lo) / h**2
                                                ),
                                            )
                                        )
                    except (ValueError, ArithmeticError, RuntimeError) as error:
                        failures.append(
                            dict(
                                radius=radius,
                                case=case,
                                maturity=time,
                                kind=kind,
                                message=str(error),
                            )
                        )
        values = pd.DataFrame(observations)
        comparisons = []
        keys = ["radius", "expire_date", "kind", "log_strike", "spot_log_change"]
        for name, other, reference in (
            ("space", "space_coarse", "space_fine"),
            ("time", "space_fine", "time_fine"),
            ("finer_space", "time_fine", "space_finer"),
            ("shift", "half_cell_shift", "space_finer"),
            ("domain", "domain_wide", "space_finer"),
        ):
            if values.empty:
                continue
            paired = values[values.case.eq(other)].merge(
                values[values.case.eq(reference)],
                on=keys,
                suffixes=("", "_ref"),
                validate="one_to_one",
            )
            paired["check"] = name
            for field in ("price", "delta", "gamma"):
                paired[field + "_change"] = paired[field] - paired[field + "_ref"]
            comparisons.append(
                paired[keys + ["check", "price_change", "delta_change", "gamma_change"]]
            )
        blend = pd.DataFrame()
        if not values.empty and 0.0 in s.radii:
            key = keys[1:]
            finest = values[values.case.eq("space_finer")]
            blend = finest.merge(
                finest[finest.radius.eq(0)],
                on=key,
                suffixes=("", "_raw"),
                validate="many_to_one",
            )
            for field in ("price", "delta", "gamma"):
                blend[field + "_blend_change"] = blend[field] - blend[field + "_raw"]
        bump_table = pd.DataFrame(bumps)
        if not bump_table.empty:
            for field in ("delta", "gamma"):
                bump_table[field + "_difference"] = (
                    bump_table["fd_" + field] - bump_table[field]
                )
        return dict(
            panel_contracts=panel,
            greek_panel=values,
            greek_refinement=(
                pd.concat(comparisons, ignore_index=True)
                if comparisons
                else pd.DataFrame()
            ),
            greek_spot_bumps=bump_table,
            blend_sensitivity=blend,
            validation_failures=pd.DataFrame(failures),
        )


@dataclass(frozen=True)
class CalibrationVegaSettings:
    bumps: tuple = (0.0001, 0.0002, 0.0005)
    log_strikes: tuple = (-0.02, 0.0, 0.02)
    intervals: tuple = (12000, 24000)
    steps_per_day: int = 128
    width: float = 0.75
    radius: float = 0.0005

    def __post_init__(self):
        if (
            not len(self.bumps)
            or not np.isfinite(self.bumps).all()
            or min(self.bumps) <= 0
            or len(set(self.bumps)) != len(self.bumps)
            or not len(self.log_strikes)
            or not np.isfinite(self.log_strikes).all()
            or len(set(self.log_strikes)) != len(self.log_strikes)
            or len(self.intervals) != 2
            or self.intervals[1] != 2 * self.intervals[0]
            or any(not isinstance(n, int) or n < 100 or n % 2 for n in self.intervals)
            or not isinstance(self.steps_per_day, int)
            or self.steps_per_day < 1
            or not np.isfinite([self.width, self.radius]).all()
            or self.width <= 0
            or self.radius < 0
        ):
            raise ValueError("Use positive unique IV bumps and two-to-one PDE grids.")


class CalibrationVegaValidator:
    """Parallel market-IV shocks, AH recalibration and conditional PDE repricing.

    Midpoints without a positive IV remain unchanged on both shock sides.
    This measures a declared calibration direction, not a model-free vega.
    """

    def __init__(self, calibration, settings=None):
        self.calibration = dict(calibration)
        self.settings = settings or CalibrationVegaSettings()

    def shocked_quotes(self, quotes, carry, bump):
        result = quotes.copy(deep=True)
        result["shock_supported"] = False
        result["shock_decimal_iv"] = 0.0
        limit = max(self.settings.bumps)
        for row in carry.to_dict("records"):
            mask = result.expire_date.eq(row["expire_date"])
            k = result.loc[mask, "strike"].to_numpy()
            f, d, t = row["forward"], row["discount_factor"], row["maturity_years"]
            intrinsic = np.maximum(f - k, 0)
            w, status = invert_time_value(
                np.log(k / f),
                (result.loc[mask, "call_mid"].to_numpy() / d - intrinsic) / f,
            )
            sigma = np.sqrt(w / t)
            supported = (status == "ready") & np.isfinite(sigma) & (sigma > limit)
            active = result.index[mask][supported]
            prices = BlackPricer(f, d, t).price(k[supported], sigma[supported] + bump)
            shift = prices - result.loc[active, "call_mid"].to_numpy()
            for field in ("call_mid", "call_bid", "call_ask"):
                if field in result:
                    result.loc[active, field] += shift
            result.loc[active, "shock_supported"] = True
            result.loc[active, "shock_decimal_iv"] = bump
        return result

    def _fit(self, quotes, carry, progress):
        c = dict(self.calibration)
        grid = AHGrid(c.pop("width"), c.pop("intervals"))
        fitter = AndreasenHugeCalibrator(grid, **c)
        arrays = [
            quotes.loc[
                quotes.expire_date.eq(row.expire_date),
                ["strike", "call_mid", "call_half_width"],
            ].to_numpy()
            for row in carry.itertuples()
        ]
        return fitter.calibrate(
            float(carry.spot.iloc[0]),
            carry.maturity_years.to_numpy(),
            carry.forward.to_numpy(),
            carry.discount_factor.to_numpy(),
            arrays,
            progress=lambda i, t: progress(f"fit pillar {i+1}: {365*t:.4g}d..."),
        )

    def run(self, saved_model, quotes, carry, progress=print):
        s = self.settings
        carry = carry.sort_values("maturity_years").reset_index(drop=True)
        panel = contract_panel(saved_model, carry, s.log_strikes)
        outputs, calibrations, failures, quote_shocks, sensitivities = (
            [],
            [],
            [],
            [],
            [],
        )
        models = {}
        shocks = [0.0] + [side * b for b in s.bumps for side in (-1, 1)]
        for bump in shocks:
            progress(f"parallel IV shock {bump:g}; recalibrating...")
            shocked = self.shocked_quotes(quotes, carry, bump)
            quote_shocks.append(shocked.assign(bump=bump))
            try:
                model, reports = self._fit(shocked, carry, progress)
                models[bump] = model
                for i, report in enumerate(reports):
                    calibrations.append(
                        dict(
                            report,
                            bump=bump,
                            pillar=i,
                            maturity=float(model.maturities[i]),
                        )
                    )
            except (ValueError, ArithmeticError, RuntimeError) as error:
                failures.append(
                    dict(bump=bump, stage="calibration", message=str(error))
                )
                continue
            try:
                variance = AHShortEndVariance(model, s.radius)
                for n in s.intervals:
                    solver = AHBackwardPricer(
                        variance,
                        domain_width=s.width,
                        space_intervals=n,
                        steps_per_day=s.steps_per_day,
                        early_time_power=3,
                    )
                    for (time, kind), group in panel.groupby(
                        ["maturity", "kind"], sort=True
                    ):
                        grids = solver.solve_many(
                            group.strike.to_numpy(), time, kind=kind
                        )
                        for contract, grid in zip(group.to_dict("records"), grids):
                            native = float(model.call_price(contract["strike"], time))
                            original = float(
                                saved_model.call_price(contract["strike"], time)
                            )
                            if kind == "put":
                                parity = float(
                                    model.discount_factor(time)
                                    * (model.forward(time) - contract["strike"])
                                )
                                native -= parity
                                original -= parity
                            price = float(grid.price(saved_model.spot))
                            if not np.isfinite([native, original, price]).all():
                                raise ArithmeticError("Nonfinite sensitivity price.")
                            outputs.append(
                                dict(
                                    contract,
                                    bump=bump,
                                    intervals=n,
                                    pde_price=price,
                                    native_price=native,
                                    saved_native_price=original,
                                )
                            )
            except (ValueError, ArithmeticError, RuntimeError) as error:
                failures.append(dict(bump=bump, stage="PDE", message=str(error)))
        prices = pd.DataFrame(outputs)
        if not prices.empty:
            keys = ["expire_date", "kind", "log_strike", "intervals"]
            for bump in s.bumps:
                pair = prices[prices.bump.eq(bump)].merge(
                    prices[prices.bump.eq(-bump)],
                    on=keys,
                    suffixes=("_up", "_down"),
                    validate="one_to_one",
                )
                for row in pair.to_dict("records"):
                    native = (row["native_price_up"] - row["native_price_down"]) / (
                        2 * bump
                    )
                    pde = (row["pde_price_up"] - row["pde_price_down"]) / (2 * bump)
                    sensitivities.append(
                        dict(
                            {k: row[k] for k in keys},
                            bump=bump,
                            native_vega=native,
                            pde_vega=pde,
                            vega_difference=pde - native,
                            native_per_vol_point=0.01 * native,
                            pde_per_vol_point=0.01 * pde,
                        )
                    )
        sensitivity = pd.DataFrame(sensitivities)
        refinement = pd.DataFrame()
        if not sensitivity.empty:
            keys = ["expire_date", "kind", "log_strike", "bump"]
            refinement = sensitivity[sensitivity.intervals.eq(s.intervals[0])].merge(
                sensitivity[sensitivity.intervals.eq(s.intervals[1])],
                on=keys,
                suffixes=("_coarse", "_fine"),
                validate="one_to_one",
            )
            refinement["pde_vega_grid_change"] = (
                refinement.pde_vega_coarse - refinement.pde_vega_fine
            )
        baseline = (
            prices[prices.bump.eq(0)].copy() if not prices.empty else pd.DataFrame()
        )
        if not baseline.empty:
            baseline["recalibration_price_change"] = (
                baseline.native_price - baseline.saved_native_price
            )
            baseline["pde_native_price_difference"] = (
                baseline.pde_price - baseline.native_price
            )
        self.models = models
        return dict(
            vega_contracts=panel,
            vega_prices=prices,
            calibration_vega=sensitivity,
            vega_refinement=refinement,
            baseline_recalibration=baseline,
            calibration_reports=pd.DataFrame(calibrations),
            quote_shocks=pd.concat(quote_shocks, ignore_index=True),
            validation_failures=pd.DataFrame(failures),
        )


class SavedDecisionRefinement:
    """Original production grid, double space resolution, and half-cell shift."""

    def run(self, saved, date, root="UNKNOWN", progress=print):
        model = saved.model(date, root)
        data = saved.dated_table(date, "decisions")
        calls = data.loc[
            data.root.eq(root) & data.kind.eq("call") & data.ah_status.eq("ready")
        ]
        if calls.empty:
            raise ValueError("No ready saved AH call decisions.")
        expiries = calls[["expire_date", "assumed_maturity_years"]].drop_duplicates()
        selected = []
        for target in (21, 35, 45):
            expiry = expiries.iloc[
                (365 * expiries.assumed_maturity_years - target).abs().argmin()
            ].expire_date
            group = calls.loc[calls.expire_date.eq(expiry)]
            for y in (-0.02, 0.0, 0.02):
                selected.append(
                    group.iloc[(group.forward_log_moneyness - y).abs().argmin()]
                )
        panel = (
            pd.DataFrame(selected)
            .drop_duplicates("entry_id")
            .sort_values(["expire_date", "strike"])
        )
        settings = saved.freeze["identity"]["config"]["settings"]["panel"]
        variance = AHShortEndVariance(model, settings["radius"])
        width, n, steps = (
            settings["width"],
            settings["intervals"],
            settings["steps_per_day"],
        )
        rows = []
        for case, intervals, shift in (
            ("production", n, 0),
            ("double_space", 2 * n, 0),
            ("half_cell_shift", 2 * n, width / (2 * n)),
        ):
            solver = AHBackwardPricer(
                variance,
                domain_width=width,
                space_intervals=intervals,
                steps_per_day=steps,
                early_time_power=3,
                state_shift=shift,
            )
            for time, group in panel.groupby("assumed_maturity_years", sort=True):
                progress(f"{date} {case}: {365*time:.4g}d, {len(group)} saved calls...")
                grids = solver.solve_many(group.strike.to_numpy(), float(time))
                for record, grid in zip(group.to_dict("records"), grids):
                    value = grid.greeks(model.spot)
                    row = {
                        name: record[name]
                        for name in (
                            "entry_id",
                            "expire_date",
                            "strike",
                            "kind",
                            "assumed_maturity_years",
                            "black_delta",
                        )
                    }
                    row.update(
                        quote_date=date,
                        root=root,
                        case=case,
                        intervals=intervals,
                        steps_per_day=steps,
                        radius=settings["radius"],
                    )
                    for field in ("price", "delta", "gamma"):
                        row[field] = float(getattr(value, field))
                        row["saved_" + field] = float(record["ah_" + field])
                        row[field + "_minus_saved"] = row[field] - row["saved_" + field]
                    rows.append(row)
        values = pd.DataFrame(rows)
        keys = ["entry_id", "quote_date", "root", "expire_date", "strike", "kind"]
        comparisons = []
        for name, reference in (
            ("double_space", "production"),
            ("half_cell_shift", "double_space"),
        ):
            pair = values[values.case.eq(name)].merge(
                values[values.case.eq(reference)],
                on=keys,
                suffixes=("", "_ref"),
                validate="one_to_one",
            )
            for field in ("price", "delta", "gamma"):
                pair[field + "_change"] = pair[field] - pair[field + "_ref"]
            pair["check"] = name
            comparisons.append(
                pair[keys + ["check", "price_change", "delta_change", "gamma_change"]]
            )
        return values, pd.concat(comparisons, ignore_index=True)


class GreekPanelWorkflow:
    """Own blend scheduling, checkpoints, and consistent producer pins."""

    @staticmethod
    def _run_blend_variant(study, date, root, radius, settings, output):
        from results import StudyResults

        saved = StudyResults(study)
        model = saved.model(date, root)
        carry = saved.carry(date)
        carry = carry.loc[carry.root.eq(root)]
        settings = replace(settings, radii=(radius,))
        tables = GreekPanelValidator(settings).run(
            model,
            carry,
            progress=lambda message: print(f"[{radius:g}] {message}", flush=True),
        )
        saved.verify_unchanged()
        checkpoint = Path(output) / "checkpoints" / f"radius_{radius:g}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        for name, frame in tables.items():
            frame.to_csv(checkpoint / f"{name}.csv", index=False)
        atomic_json(
            checkpoint / "audit.json",
            dict(
                radius=radius,
                settings=asdict(settings),
                input_sha256=saved.inputs,
                output_sha256={p.name: sha256(p) for p in checkpoint.glob("*.csv")},
            ),
        )
        return tables, saved.inputs

    @staticmethod
    def _merge_blend_variants(parts):
        tables = {
            name: pd.concat([part[name] for part in parts], ignore_index=True)
            for name in parts[0]
        }
        tables["panel_contracts"] = (
            tables["panel_contracts"].drop_duplicates().reset_index(drop=True)
        )
        values = tables["greek_panel"]
        if not values.empty and values.radius.eq(0).any():
            keys = ["expire_date", "kind", "log_strike", "spot_log_change"]
            finest = values[values.case.eq("space_finer")]
            blend = finest.merge(
                finest[finest.radius.eq(0)],
                on=keys,
                suffixes=("", "_raw"),
                validate="many_to_one",
            )
            for field in ("price", "delta", "gamma"):
                blend[field + "_blend_change"] = blend[field] - blend[field + "_raw"]
            tables["blend_sensitivity"] = blend
        return tables

    def __init__(self, settings=None, workers=1):
        self.settings = settings or GreekPanelSettings()
        if type(workers) is not int or not 1 <= workers <= len(self.settings.radii):
            raise ValueError("Workers must be between one and the number of radii.")
        self.workers = workers

    def run(self, saved, date, root, output):
        output = saved.safe_output(output)
        if output.exists() and any(output.iterdir()):
            raise ValueError("Use a new empty validation directory.")
        output.mkdir(parents=True, exist_ok=True)
        arguments = [
            (str(saved.folder), date, root, radius, self.settings, str(output))
            for radius in self.settings.radii
        ]
        if self.workers == 1:
            parts = [self._run_blend_variant(*args) for args in arguments]
        else:
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                futures = [
                    pool.submit(self._run_blend_variant, *args) for args in arguments
                ]
                parts = [future.result() for future in futures]
        for _, pins in parts:
            for path, digest in pins.items():
                if path in saved.inputs and saved.inputs[path] != digest:
                    raise ValueError("Workers consumed inconsistent input versions.")
                saved.inputs[path] = digest
        saved.verify_unchanged()
        return self._merge_blend_variants([tables for tables, _ in parts])
