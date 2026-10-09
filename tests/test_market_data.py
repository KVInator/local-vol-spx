import unittest
import pandas as pd
from market_data import AuditSettings, HedgingDataAudit


def quote_audit_quote(day="2023-09-01", strike=100.0, expiry="2023-10-06", spot=100.0):
    stamp = pd.Timestamp(f"{day} 16:00", tz="America/New_York")
    discount, forward = (0.995, 100.5)
    cp = discount * (forward - strike)
    put_mid = 20.0
    return {
        "quote_unixtime": stamp.timestamp(),
        "quote_readtime": f"{day} 16:00",
        "quote_date": day,
        "expire_date": expiry,
        "underlying_last": spot,
        "strike": strike,
        "c_bid": put_mid + cp - 0.1,
        "c_ask": put_mid + cp + 0.1,
        "p_bid": put_mid - 0.1,
        "p_ask": put_mid + 0.1,
    }


class TestHedgingDataAudit(unittest.TestCase):

    def run_audit(self, rows, **settings):
        return HedgingDataAudit(AuditSettings(**settings)).run(pd.DataFrame(rows))

    def test_exact_parity_and_bracket_headers(self):
        raw = pd.DataFrame(
            [quote_audit_quote(strike=k) for k in (98, 99, 100, 101, 102)]
        )
        raw.columns = [f" [{c.upper()}] " for c in raw.columns]
        result = HedgingDataAudit().run(raw)
        parity = result["parity_summary"].iloc[0]
        self.assertEqual(parity["status"], "diagnostic_fit")
        self.assertAlmostEqual(parity["fitted_discount"], 0.995, places=10)
        self.assertAlmostEqual(parity["fitted_forward"], 100.5, places=10)
        self.assertEqual(parity["outside_parity_bands"], 0)
        self.assertTrue(
            result["quote_panel"]["identity_status"].eq("provisional_no_root").all()
        )
        self.assertTrue(
            result["expiry_metadata"]["model_fixing_timestamp_utc"].eq("").all()
        )

    def test_duplicate_quarantine_does_not_choose_first(self):
        first, duplicate = (quote_audit_quote(), quote_audit_quote())
        duplicate["c_ask"] += 1
        result = self.run_audit([first, duplicate, quote_audit_quote("2023-09-05")])
        self.assertEqual(result["row_audit"]["duplicate_close_key"].sum(), 2)
        self.assertFalse(result["row_audit"].iloc[:2]["c_mark_usable"].any())
        self.assertEqual(len(result["quote_panel"]), 2)

    def test_root_separates_otherwise_identical_keys(self):
        rows = [dict(quote_audit_quote(), option_root=root) for root in ("SPX", "SPXW")]
        result = self.run_audit(rows, root_column="option_root")
        self.assertFalse(result["row_audit"]["duplicate_close_key"].any())
        self.assertEqual(len(result["contract_continuity"]), 4)
        self.assertTrue(
            result["expiry_metadata"]["settlement_status"].eq("unverified").all()
        )

    def test_crossed_locked_zero_and_missing_quotes(self):
        rows = [quote_audit_quote(strike=k) for k in (98, 99, 100, 101)]
        rows[0]["c_bid"], rows[0]["c_ask"] = (2, 1)
        rows[1]["c_bid"], rows[1]["c_ask"] = (1, 1)
        rows[2]["c_bid"], rows[2]["c_ask"] = (0, 0.1)
        rows[3]["c_bid"] = "broken"
        result = self.run_audit(rows)
        calls = result["quote_panel"].query("kind == 'call'").set_index("strike")
        self.assertTrue(calls.loc[98, "bad_quote"])
        self.assertFalse(calls.loc[99, "mark_usable"])
        self.assertTrue(calls.loc[100, "mark_usable"])
        self.assertFalse(calls.loc[100, "two_sided_quote"])
        self.assertTrue(calls.loc[101, "bad_quote"])

    def test_timestamp_mismatch_and_bad_epoch(self):
        bad_read = quote_audit_quote(strike=99)
        bad_epoch = quote_audit_quote(strike=100)
        bad_read["quote_readtime"] = "2023-09-01 15:59"
        bad_epoch["quote_unixtime"] = "not a number"
        result = self.run_audit([bad_read, bad_epoch, quote_audit_quote(strike=101)])
        self.assertTrue(result["row_audit"].iloc[0]["timestamp_mismatch"])
        self.assertTrue(result["row_audit"].iloc[1]["bad_timestamp"])
        self.assertFalse(result["row_audit"].iloc[:2]["c_mark_usable"].any())

    def test_missing_close_stays_in_continuity_calendar(self):
        off_close = quote_audit_quote("2023-09-05")
        off_close["quote_unixtime"] += 60
        off_close["quote_readtime"] = "2023-09-05 16:01"
        result = self.run_audit(
            [quote_audit_quote(), off_close, quote_audit_quote("2023-09-06")]
        )
        self.assertTrue(
            result["contract_continuity"]["missing_between_first_and_last"].eq(1).all()
        )
        first = result["adjacent_observations"]
        first = first[first["quote_date"].eq(pd.Timestamp("2023-09-01"))]
        self.assertTrue(first["status"].eq("missing_or_ambiguous_next").all())
        self.assertTrue(first["next_date"].eq(pd.Timestamp("2023-09-05")).all())

    def test_expiry_is_not_missing_and_no_strike_substitution(self):
        rows = [
            quote_audit_quote(expiry="2023-09-05"),
            quote_audit_quote(strike=99),
            quote_audit_quote("2023-09-05", strike=98),
        ]
        result = self.run_audit(rows)
        first = result["adjacent_observations"]
        first = first[first["quote_date"].eq(pd.Timestamp("2023-09-01"))]
        expiring = first[first["expire_date"].eq(pd.Timestamp("2023-09-05"))]
        self.assertTrue(expiring["status"].eq("expiry_before_or_on_next_date").all())
        alive = first[first["strike"].eq(99)]
        self.assertTrue(alive["status"].eq("missing_or_ambiguous_next").all())
        self.assertTrue(alive["next_mid"].isna().all())

    def test_scope_exit_and_vendor_time_are_explicit(self):
        first = quote_audit_quote()
        second = quote_audit_quote("2023-09-05", spot=90)
        first["expire_unix"] = pd.Timestamp("2023-10-06 13:30", tz="UTC").timestamp()
        result = self.run_audit([first, second])
        pairs = result["adjacent_observations"]
        pairs = pairs[pairs["quote_date"].eq(pd.Timestamp("2023-09-01"))]
        self.assertTrue(pairs["status"].eq("out_of_scope_next").all())
        self.assertTrue(
            result["expiry_metadata"]["settlement_status"].eq("unverified").all()
        )
        self.assertTrue(
            result["contract_continuity"]["usable_observations"].eq(2).all()
        )

    def test_inconsistent_spot_quarantines_snapshot(self):
        result = self.run_audit(
            [quote_audit_quote(strike=99), quote_audit_quote(strike=100, spot=101)]
        )
        self.assertTrue(result["row_audit"]["spot_inconsistent"].all())
        self.assertFalse(result["row_audit"]["c_mark_usable"].any())

    def test_missing_schema_fails_before_outputs(self):
        raw = pd.DataFrame([quote_audit_quote()]).drop(columns="p_ask")
        with self.assertRaisesRegex(ValueError, "Missing raw columns"):
            HedgingDataAudit().run(raw)


import numpy as np
from market_data import QuotePolicy, QuotePolicySettings


def quote_policy_policy(start="2023-08-01", end="2024-01-31"):
    dates = pd.bdate_range(start, end)
    return pd.DataFrame(
        {
            "quote_date": dates,
            "reference_session": True,
            "early_cash_close": False,
            "session_policy": "provisional_standard_session",
        }
    )


def quote_policy_quote(day="2023-09-01", expiry="2023-09-29", strike=100.0, **changes):
    stamp = pd.Timestamp(f"{day} 16:00", tz="America/New_York")
    row = {
        "quote_unixtime": stamp.timestamp(),
        "quote_readtime": f"{day} 16:00",
        "quote_date": day,
        "expire_date": expiry,
        "underlying_last": 100.0,
        "strike": strike,
        "c_bid": 5.0,
        "c_ask": 5.2,
        "p_bid": 4.0,
        "p_ask": 4.2,
    }
    row.update(changes)
    return row


class QuotePolicyTests(unittest.TestCase):

    def prepare(self, rows, p=None, settings=None):
        return QuotePolicy(quote_policy_policy() if p is None else p, settings).prepare(
            pd.DataFrame(rows)
        )

    def test_monthly_aliases_are_conservatively_excluded(self):
        rows = [
            quote_policy_quote(expiry=d)
            for d in ("2023-09-14", "2023-09-15", "2023-09-16", "2023-09-29")
        ]
        tables, audit = self.prepare(rows)
        self.assertEqual(len(tables["pilot_quotes"]), 2)
        self.assertTrue(
            tables["pilot_quotes"]["expire_date"].eq(pd.Timestamp("2023-09-29")).all()
        )
        self.assertIn("2023-09-14", audit["monthly_date_aliases_excluded"])
        self.assertFalse(audit["contract_identity_verified"])

    def test_holiday_adjusted_monthly_dates(self):
        p = quote_policy_policy("2022-03-01", "2022-07-01")
        mask = p["quote_date"].eq(pd.Timestamp("2022-04-15"))
        p.loc[mask, "reference_session"] = False
        p.loc[mask, "session_policy"] = "nonreference_date"
        s = QuotePolicySettings(start="2022-04-01", end="2022-04-01")
        model = QuotePolicy(p, s)
        expected = pd.to_datetime(
            ["2022-04-13", "2022-04-14", "2022-04-15", "2022-04-16"]
        )
        self.assertTrue(expected.isin(model.monthly_aliases).all())
        tables, _ = model.prepare(
            pd.DataFrame(
                [
                    quote_policy_quote(day="2022-04-01", expiry="2022-04-14"),
                    quote_policy_quote(day="2022-04-01", expiry="2022-04-29"),
                ]
            )
        )
        self.assertEqual(len(tables["pilot_quotes"]), 2)

    def test_future_observation_availability_does_not_select_entries(self):
        baseline, _ = self.prepare([quote_policy_quote()])
        p = quote_policy_policy()
        mask = p["quote_date"].eq(pd.Timestamp("2023-09-29"))
        p.loc[mask, "session_policy"] = "missing_observation"
        missing, _ = self.prepare([quote_policy_quote()], p)
        pd.testing.assert_frame_equal(baseline["pilot_quotes"], missing["pilot_quotes"])
        self.assertTrue(missing["pilot_quotes"]["entry_research_candidate"].all())

    def test_quote_quality_is_side_specific_and_zero_bids_are_bounds(self):
        rows = [
            quote_policy_quote(strike=99, c_bid=6.0, c_ask=5.0),
            quote_policy_quote(strike=100, c_bid=0.0, c_ask=0.2),
            quote_policy_quote(strike=101, c_bid=1.0, c_ask=1.0),
        ]
        tables, _ = self.prepare(rows)
        self.assertEqual(tables["pilot_quotes"]["kind"].tolist(), ["put", "put", "put"])
        self.assertEqual(tables["zero_bid_bounds"]["kind"].tolist(), ["call"])
        self.assertFalse(tables["zero_bid_bounds"]["entry_research_candidate"].any())

    def test_nullable_string_input_with_missing_side(self):
        raw = pd.DataFrame(
            [quote_policy_quote(c_bid=None), quote_policy_quote(strike=101)]
        ).astype("string")
        tables, _ = QuotePolicy(quote_policy_policy()).prepare(raw)
        quotes = tables["pilot_quotes"]
        self.assertEqual(len(quotes), 3)
        self.assertEqual(quotes.query("kind == 'put'")["strike"].tolist(), [100, 101])

    def test_duplicate_rows_are_all_quarantined(self):
        tables, audit = self.prepare(
            [
                quote_policy_quote(),
                quote_policy_quote(c_ask=6),
                quote_policy_quote(strike=102),
            ]
        )
        self.assertEqual(audit["duplicate_rows_quarantined"], 2)
        self.assertTrue(tables["pilot_quotes"]["strike"].eq(102).all())

    def test_calendar_and_current_session_exclusions(self):
        p = quote_policy_policy()
        mask = p["quote_date"].eq(pd.Timestamp("2023-09-05"))
        p.loc[mask, "session_policy"] = "missing_observation"
        mask = p["quote_date"].eq(pd.Timestamp("2023-10-06"))
        p.loc[mask, "early_cash_close"] = True
        tables, _ = self.prepare(
            [
                quote_policy_quote(),
                quote_policy_quote(day="2023-09-05"),
                quote_policy_quote(expiry="2023-10-06"),
            ],
            p,
        )
        self.assertEqual(len(tables["pilot_quotes"]), 2)
        statuses = tables["quote_selection_summary"]["quote_status"].tolist()
        self.assertIn("session_excluded", statuses)
        self.assertIn("expiry_calendar_excluded", statuses)
        row = tables["session_timeline"].set_index("quote_date").loc["2023-09-05"]
        self.assertEqual(row["session_policy"], "missing_observation")

    def test_elapsed_utc_maturity_handles_dst(self):
        s = QuotePolicySettings(start="2023-11-03", end="2023-11-03")
        tables, _ = self.prepare(
            [quote_policy_quote(day="2023-11-03", expiry="2023-11-10")], settings=s
        )
        maturity = tables["pilot_quotes"]["assumed_maturity_years"].iloc[0]
        self.assertAlmostEqual(maturity, 169 / (365 * 24))

    def test_wide_calibration_coverage_does_not_force_entry_selection(self):
        tables, _ = self.prepare([quote_policy_quote(strike=100 * np.exp(0.06))])
        self.assertEqual(len(tables["pilot_quotes"]), 2)
        self.assertFalse(tables["pilot_quotes"]["entry_research_candidate"].any())
        self.assertTrue(tables["pilot_quotes"]["root"].eq("UNKNOWN").all())

    def test_policy_dates_and_horizon_are_checked(self):
        p = quote_policy_policy()
        with self.assertRaisesRegex(ValueError, "unique"):
            QuotePolicy(pd.concat([p, p.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "horizon"):
            QuotePolicy(p.loc[p["quote_date"] < "2023-11-01"])
        p["reference_session"] = p["reference_session"].astype(object)
        p.loc[0, "reference_session"] = "unknown"
        with self.assertRaisesRegex(ValueError, "flags"):
            QuotePolicy(p)


from carry import CarrySettings, CarryEstimator


def carry_estimation_panel(
    discount=None, forward=100.35, day="2023-09-01", spread_half=0.02
):
    timestamp = pd.Timestamp(f"{day} 16:00", tz="America/New_York").tz_convert("UTC")
    fixing = pd.Timestamp("2023-09-29 16:00", tz="America/New_York").tz_convert("UTC")
    maturity = (fixing - timestamp).total_seconds() / (365 * 86400)
    discount = np.exp(-0.05 * maturity) if discount is None else discount
    rows = []
    for strike in np.linspace(97.5, 102.5, 21):
        for kind, mid in [("call", 5 + discount * (forward - strike)), ("put", 5)]:
            rows.append(
                {
                    "quote_date": day,
                    "root": "UNKNOWN",
                    "expire_date": "2023-09-29",
                    "strike": strike,
                    "kind": kind,
                    "bid": mid - spread_half,
                    "ask": mid + spread_half,
                    "underlying_last": 100.0,
                    "quote_timestamp_utc": timestamp,
                    "assumed_fixing_utc": fixing,
                    "assumed_maturity_years": maturity,
                    "quote_status": "research_calibration_quote",
                }
            )
    return pd.DataFrame(rows)


class CarryEstimationTests(unittest.TestCase):

    def run_panel(self, q):
        return CarryEstimator(CarrySettings(windows=(0.03,))).run(q)

    def test_exact_forward_and_discount_recovery(self):
        q = carry_estimation_panel()
        tables, audit = self.run_panel(q)
        row = tables["carry_estimates"].iloc[0]
        expected = np.exp(-0.05 * q["assumed_maturity_years"].iloc[0])
        self.assertEqual(row["status"], "fitted")
        self.assertAlmostEqual(row["discount_factor"], expected, places=10)
        self.assertAlmostEqual(row["forward"], 100.35, places=10)
        self.assertAlmostEqual(row["implied_zero_rate_pct"], 5, places=8)
        self.assertTrue(row["original_bands_feasible"])
        self.assertLess(row["minimum_band_multiplier"], 1e-06)
        self.assertFalse(audit["daily_discount_curve_verified"])

    def test_known_fixed_rate_scenario_recovers_forward(self):
        tables, _ = self.run_panel(carry_estimation_panel())
        row = tables["fixed_rate_sensitivity"].query("annual_rate_pct == 5").iloc[0]
        self.assertAlmostEqual(row["forward"], 100.35, places=10)
        self.assertLess(row["rms_parity_half_widths"], 1e-08)

    def test_unpaired_side_remains_out_of_parity_fit(self):
        q = carry_estimation_panel()
        q = q.loc[~(q["strike"].eq(100) & q["kind"].eq("put"))]
        tables, audit = self.run_panel(q)
        self.assertEqual(audit["matched_pairs"], 20)
        self.assertEqual(audit["unpaired_quote_sides"], 1)
        self.assertAlmostEqual(
            tables["carry_estimates"].iloc[0]["forward"], 100.35, places=10
        )

    def test_duplicate_sides_are_not_averaged(self):
        q = carry_estimation_panel()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_panel(pd.concat([q, q.iloc[:1]]))

    def test_pairs_never_cross_quote_dates(self):
        q = pd.concat(
            [
                carry_estimation_panel().query("kind == 'call'"),
                carry_estimation_panel(day="2023-09-05").query("kind == 'put'"),
            ]
        )
        tables, audit = self.run_panel(q)
        self.assertEqual(audit["matched_pairs"], 0)
        self.assertTrue(
            tables["carry_estimates"]["status"].eq("insufficient_pairs").all()
        )

    def test_negative_implied_rates_are_not_clipped(self):
        tables, _ = self.run_panel(carry_estimation_panel(discount=1.003))
        row = tables["carry_estimates"].iloc[0]
        self.assertAlmostEqual(row["discount_factor"], 1.003, places=10)
        self.assertLess(row["implied_zero_rate_pct"], 0)
        self.assertEqual(row["status"], "fitted")

    def test_discount_identification_worsens_with_wider_bands(self):
        narrow, _ = self.run_panel(carry_estimation_panel(spread_half=0.02))
        wide, _ = self.run_panel(carry_estimation_panel(spread_half=0.2))
        a = narrow["carry_estimates"].iloc[0]
        b = wide["carry_estimates"].iloc[0]
        self.assertGreater(
            b["discount_upper"] - b["discount_lower"],
            5 * (a["discount_upper"] - a["discount_lower"]),
        )

    def test_incompatible_parity_bands_are_reported(self):
        q = carry_estimation_panel()
        mask = q["strike"].eq(100) & q["kind"].eq("call")
        q.loc[mask, ["bid", "ask"]] += 0.5
        tables, _ = self.run_panel(q)
        row = tables["carry_estimates"].iloc[0]
        self.assertFalse(row["original_bands_feasible"])
        self.assertGreater(row["minimum_band_multiplier"], 1)
        self.assertEqual(row["band_bounds_status"], "original_bands_infeasible")
        self.assertTrue(pd.isna(row["rate_band_width_pp"]))
        self.assertLess(abs(row["forward"] - 100.35), 0.03)

    def test_invalid_discount_is_not_promoted(self):
        q = carry_estimation_panel()
        mask = q["kind"].eq("call")
        mid = 5 + 0.5 * (q.loc[mask, "strike"] - 100)
        q.loc[mask, "bid"] = mid - 0.02
        q.loc[mask, "ask"] = mid + 0.02
        tables, _ = self.run_panel(q)
        self.assertEqual(
            tables["carry_estimates"].iloc[0]["status"], "invalid_discount_or_forward"
        )

    def test_narrow_window_is_not_silently_widened(self):
        tables, _ = CarryEstimator(CarrySettings(windows=(0.001, 0.03))).run(
            carry_estimation_panel()
        )
        fits = tables["carry_estimates"].set_index("window")
        self.assertEqual(fits.loc[0.001, "status"], "insufficient_pairs")
        self.assertEqual(fits.loc[0.03, "status"], "fitted")
        self.assertTrue(
            pd.isna(tables["window_sensitivity"].iloc[0]["forward_window_range_points"])
        )

    def test_maturity_and_snapshot_conflicts_fail(self):
        q = carry_estimation_panel()
        q.loc[0, "assumed_maturity_years"] += 0.01
        with self.assertRaisesRegex(ValueError, "maturity"):
            self.run_panel(q)
        q = carry_estimation_panel()
        q.loc[0, "underlying_last"] = 101
        with self.assertRaisesRegex(ValueError, "snapshot"):
            self.run_panel(q)


from carry import CarryInputSettings, CarryInputs


def carry_inputs_panel(day="2023-09-01", expiry="2023-09-29", forward=100.35):
    stamp = pd.Timestamp(f"{day} 16:00", tz="America/New_York").tz_convert("UTC")
    fixing = pd.Timestamp(f"{expiry} 16:00", tz="America/New_York").tz_convert("UTC")
    time = (fixing - stamp).total_seconds() / (365 * 86400)
    discount = np.exp(-0.05 * time)
    rows = []
    for strike in np.linspace(97.5, 102.5, 21):
        for kind, mid in [("call", 5 + discount * (forward - strike)), ("put", 5)]:
            rows.append(
                dict(
                    quote_date=day,
                    root="UNKNOWN",
                    expire_date=expiry,
                    strike=strike,
                    kind=kind,
                    bid=mid - 0.02,
                    ask=mid + 0.02,
                    underlying_last=100.0,
                    quote_timestamp_utc=stamp,
                    assumed_fixing_utc=fixing,
                    assumed_maturity_years=time,
                    quote_status="research_calibration_quote",
                )
            )
    return pd.DataFrame(rows)


class CarryInputTests(unittest.TestCase):

    def test_discount_units_and_conditional_forward_refits(self):
        q = carry_inputs_panel()
        tables, _ = CarryInputs().prepare(q)
        time = q.assumed_maturity_years.iloc[0]
        for _, row in tables["carry_inputs"].iterrows():
            expected_d = np.exp(-row.annual_rate * time)
            expected_f = 100 + np.exp(-0.05 * time) / expected_d * 0.35
            self.assertAlmostEqual(row.discount_factor, expected_d, places=12)
            self.assertAlmostEqual(row.forward, expected_f, places=10)
        self.assertEqual(tables["primary_carry"].case.iloc[0], "rate_5pct")

    def test_otm_selection_and_put_conversion(self):
        tables, _ = CarryInputs().prepare(carry_inputs_panel())
        q = tables["calibration_quotes"]
        self.assertEqual(len(q), 21)
        self.assertFalse(q.duplicated(CarryInputs.KEY).any())
        self.assertTrue(q.loc[q.strike.lt(q.forward), "source_kind"].eq("put").all())
        self.assertTrue(q.loc[q.strike.ge(q.forward), "source_kind"].eq("call").all())
        adjustment = np.where(
            q.source_kind.eq("put"), q.discount_factor * (q.forward - q.strike), 0
        )
        np.testing.assert_allclose(q.call_bid, q.source_bid + adjustment)
        np.testing.assert_allclose(q.call_half_width, (q.source_ask - q.source_bid) / 2)

    def test_missing_preferred_side_uses_flagged_fallback(self):
        q = carry_inputs_panel()
        q = q.loc[~(q.kind.eq("put") & q.strike.eq(99))]
        tables, _ = CarryInputs().prepare(q)
        selected = tables["calibration_quotes"]
        row = selected.loc[selected.strike.eq(99)].iloc[0]
        self.assertEqual(row.source_kind, "call")
        self.assertTrue(row.preferred_side_missing)
        self.assertEqual(len(selected), 21)
        self.assertEqual(selected.strike.min(), 97.5)
        self.assertEqual(selected.strike.max(), 102.5)

    def test_incompatible_parity_is_retained(self):
        q = carry_inputs_panel()
        q.loc[q.kind.eq("call") & q.strike.eq(100), ["bid", "ask"]] += 0.5
        tables, _ = CarryInputs().prepare(q)
        row = tables["primary_carry"].iloc[0]
        self.assertTrue(row.carry_ready)
        self.assertTrue(row.parity_bands_incompatible)
        self.assertEqual(row.carry_status, "ready_with_parity_incompatibility")
        self.assertEqual(len(tables["calibration_quotes"]), 21)

    def test_insufficient_pairs_preserve_group_without_forward(self):
        q = carry_inputs_panel().loc[lambda x: x.strike.le(98.5)]
        tables, _ = CarryInputs().prepare(q)
        carry = tables["carry_inputs"]
        self.assertEqual(len(carry), 3)
        self.assertFalse(carry.carry_ready.any())
        self.assertTrue(carry.forward.isna().all())
        self.assertTrue(carry.discount_factor.gt(0).all())
        self.assertTrue(tables["calibration_quotes"].call_mid.isna().all())

    def test_pairs_cannot_cross_dates(self):
        q = pd.concat(
            [
                carry_inputs_panel().query("kind == 'call'"),
                carry_inputs_panel(day="2023-09-05").query("kind == 'put'"),
            ],
            ignore_index=True,
        )
        tables, audit = CarryInputs().prepare(q)
        self.assertEqual(audit["matched_pairs"], 0)
        self.assertEqual(len(tables["primary_carry"]), 2)
        self.assertFalse(tables["primary_carry"].carry_ready.any())

    def test_later_quotes_cannot_change_earlier_inputs(self):
        original = carry_inputs_panel()
        first, _ = CarryInputs().prepare(original)
        augmented = pd.concat(
            [original, carry_inputs_panel(day="2023-09-05", forward=101)],
            ignore_index=True,
        )
        augmented["next_mid"] = 999999.0
        second, _ = CarryInputs().prepare(augmented)
        for name in ["primary_carry", "calibration_quotes"]:
            earlier = second[name].loc[
                second[name].quote_date.eq(pd.Timestamp("2023-09-01"))
            ]
            pd.testing.assert_frame_equal(first[name], earlier.reset_index(drop=True))

    def test_daily_snapshot_must_be_consistent_across_expiries(self):
        second = carry_inputs_panel(expiry="2023-10-06")
        second["underlying_last"] = 100.5
        with self.assertRaisesRegex(ValueError, "daily snapshot"):
            CarryInputs().prepare(
                pd.concat([carry_inputs_panel(), second], ignore_index=True)
            )

    def test_actual_utc_maturity_survives_daylight_saving_change(self):
        q = carry_inputs_panel(day="2023-11-02", expiry="2023-11-09")
        tables, _ = CarryInputs().prepare(q)
        row = tables["primary_carry"].iloc[0]
        self.assertAlmostEqual(row.maturity_years, 169 / (365 * 24), places=12)
        self.assertAlmostEqual(
            row.discount_factor, np.exp(-0.05 * 169 / (365 * 24)), places=12
        )

    def test_call_bound_breach_is_reported_without_clipping(self):
        q = carry_inputs_panel()
        q.loc[q.kind.eq("call") & q.strike.eq(102.5), ["bid", "ask"]] = [105, 106]
        tables, _ = CarryInputs().prepare(q)
        row = tables["calibration_quotes"].loc[lambda x: x.strike.eq(102.5)].iloc[0]
        self.assertTrue(row.band_disjoint_from_call_bounds)
        self.assertEqual(row.call_bid, 105)
        self.assertEqual(row.call_ask, 106)

    def test_raw_quotes_and_assumption_labels_remain_intact(self):
        q = carry_inputs_panel()
        before = q.copy(deep=True)
        _, audit = CarryInputs().prepare(q)
        pd.testing.assert_frame_equal(q, before)
        self.assertFalse(audit["funding_curve_verified"])
        self.assertFalse(audit["backtest_performed"])
        self.assertFalse(audit["prices_clipped"])

    def test_primary_rate_must_be_present(self):
        with self.assertRaises(ValueError):
            CarryInputSettings(primary_rate=0.05, rates=(0.03, 0.07))


from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from data_audit import HistoricalAuditSettings, HistoricalDataAudit


def history_quote(day, clock="16:00", strike=100, spot=100, expiry="2023-10-06"):
    stamp = pd.Timestamp(f"{day} {clock}", tz="America/New_York")
    return {
        "quote_unixtime": stamp.timestamp(),
        "quote_readtime": f"{day} {clock}",
        "quote_date": day,
        "expire_date": expiry,
        "underlying_last": spot,
        "strike": strike,
        "c_bid": 1.0,
        "c_ask": 1.2,
        "p_bid": 1.0,
        "p_ask": 1.2,
    }


class TestHistoricalDataAudit(unittest.TestCase):

    def run_case(self, monthly, start="2023-08-31", end="2023-09-06", **kwargs):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            root = Path(directory)
            files = []
            for month, rows in monthly.items():
                path = root / f"spx_eod_{month}.txt"
                pd.DataFrame(rows).to_csv(path, index=False)
                files.append(path)
            settings = HistoricalAuditSettings(start=start, end=end, **kwargs)
            return HistoricalDataAudit(settings).run(files, root / "outputs")

    def test_cross_month_pair_and_weekend_holiday_are_not_gaps(self):
        tables, _ = self.run_case(
            {
                "202308": [history_quote("2023-08-31")],
                "202309": [
                    history_quote("2023-09-01"),
                    history_quote("2023-09-05"),
                    history_quote("2023-09-06"),
                ],
            }
        )
        adjacent = tables["adjacent_summary"]
        cross = adjacent.loc[adjacent["cross_month"]]
        self.assertEqual(cross["contracts"].sum(), 2)
        self.assertTrue(cross["status"].eq("matched").all())
        self.assertEqual(
            adjacent.loc[adjacent["status"].eq("sample_end"), "contracts"].sum(), 2
        )
        self.assertTrue(tables["session_anomalies"].empty)

    def test_missing_session_is_not_skipped(self):
        tables, _ = self.run_case(
            {
                "202308": [history_quote("2023-08-31")],
                "202309": [history_quote("2023-09-05"), history_quote("2023-09-06")],
            }
        )
        a = tables["adjacent_summary"]
        first = a.loc[a["quote_date"].eq(pd.Timestamp("2023-08-31"))]
        self.assertTrue(first["next_date"].eq(pd.Timestamp("2023-09-01")).all())
        self.assertTrue(first["status"].eq("missing_session_observation").all())

    def test_early_close_clock_is_reported_without_relabeling(self):
        tables, _ = self.run_case(
            {"202307": [history_quote("2023-07-03"), history_quote("2023-07-04")]},
            start="2023-07-03",
            end="2023-07-05",
        )
        d = tables["daily_summary"].set_index("quote_date")
        early = d.loc[pd.Timestamp("2023-07-03")]
        self.assertTrue(early["early_cash_close"])
        self.assertEqual(early["clocks_ny"], "16:00:00")
        self.assertEqual(early["after_cash_close_rows"], 1)
        self.assertEqual(
            d.loc[pd.Timestamp("2023-07-04"), "status"], "observed_nonreference_date"
        )

    def test_single_holiday_window_can_be_audited(self):
        tables, _ = self.run_case(
            {"202307": [history_quote("2023-07-04")]},
            start="2023-07-04",
            end="2023-07-04",
        )
        self.assertEqual(
            tables["daily_summary"].iloc[0]["status"], "observed_nonreference_date"
        )
        self.assertTrue(tables["adjacent_summary"].empty)
        tables, _ = self.run_case(
            {"202307": [history_quote("2023-07-05")]},
            start="2023-07-04",
            end="2023-07-04",
        )
        self.assertTrue(tables["daily_summary"].empty)

    def test_missing_schema_is_recorded_as_failed_not_success(self):
        bad = history_quote("2023-09-01")
        del bad["p_ask"]
        tables, audit = self.run_case(
            {"202309": [bad]}, start="2023-09-01", end="2023-09-06"
        )
        self.assertEqual(audit["files_by_status"], {"file_failed": 1})
        self.assertIn("p_ask", tables["file_summary"].iloc[0]["missing_columns"])
        self.assertTrue(tables["daily_summary"]["status"].eq("file_failed").all())

    def test_duplicate_next_key_is_quarantined(self):
        q = history_quote("2023-09-01")
        tables, _ = self.run_case(
            {"202308": [history_quote("2023-08-31")], "202309": [q, q]},
            end="2023-09-01",
        )
        a = tables["adjacent_summary"]
        self.assertTrue(a["status"].eq("missing_or_ambiguous_next").all())
        self.assertEqual(tables["daily_summary"].iloc[-1]["duplicate_key_rows"], 2)

    def test_scope_exit_uses_wider_next_day_lookup(self):
        tables, _ = self.run_case(
            {
                "202308": [history_quote("2023-08-31")],
                "202309": [history_quote("2023-09-01", spot=90)],
            },
            end="2023-09-01",
        )
        self.assertTrue(
            tables["adjacent_summary"]["status"].eq("out_of_scope_next").all()
        )

    def test_expiry_boundary_is_not_missing_data(self):
        tables, _ = self.run_case(
            {
                "202308": [history_quote("2023-08-31", expiry="2023-09-01")],
                "202309": [history_quote("2023-09-01")],
            },
            end="2023-09-01",
        )
        a = tables["adjacent_summary"]
        first = a.loc[a["quote_date"].eq(pd.Timestamp("2023-08-31"))]
        self.assertTrue(first["status"].eq("expiry_before_or_on_next_session").all())

    def test_multiple_clocks_do_not_select_latest_snapshot(self):
        tables, _ = self.run_case(
            {
                "202309": [
                    history_quote("2023-09-01", "15:59"),
                    history_quote("2023-09-01", strike=101),
                ]
            },
            start="2023-09-01",
            end="2023-09-05",
        )
        first = tables["daily_summary"].iloc[0]
        self.assertEqual(first["snapshot_status"], "multiple_or_missing_timestamps")
        self.assertEqual(first["usable_c_marks"], 0)
        self.assertTrue(tables["adjacent_summary"].empty)

    def test_dst_and_readtime_disagreement(self):
        winter = history_quote("2023-01-03")
        summer = history_quote("2023-07-03", "13:00")
        bad = history_quote("2023-07-05")
        bad["quote_readtime"] = "2023-07-05 15:00"
        tables, _ = self.run_case(
            {"202301": [winter], "202307": [summer, bad]},
            start="2023-01-03",
            end="2023-07-05",
        )
        d = tables["daily_summary"].set_index("quote_date")
        self.assertEqual(d.loc[pd.Timestamp("2023-01-03"), "timestamp_mismatch"], 0)
        self.assertEqual(d.loc[pd.Timestamp("2023-07-03"), "at_cash_close_rows"], 1)
        self.assertEqual(d.loc[pd.Timestamp("2023-07-05"), "timestamp_mismatch"], 1)
        self.assertEqual(d.loc[pd.Timestamp("2023-07-05"), "usable_c_marks"], 0)

    def test_wrong_file_month_and_bad_quotes_are_counted(self):
        wrong = history_quote("2023-08-31")
        crossed = history_quote("2023-09-01")
        crossed["c_bid"], crossed["c_ask"] = (2, 1)
        crossed["p_bid"] = 0
        tables, _ = self.run_case(
            {"202309": [wrong, crossed]}, start="2023-09-01", end="2023-09-05"
        )
        info = tables["file_summary"].iloc[0]
        self.assertEqual(info["file_month_mismatch"], 1)
        self.assertEqual(info["c_bad_quote"], 1)
        self.assertEqual(info["p_zero_bid"], 1)

    def test_roots_separate_keys_but_do_not_certify_fixings(self):
        rows = [
            dict(history_quote("2023-09-01"), option_root=r) for r in ("SPX", "SPXW")
        ]
        tables, audit = self.run_case(
            {"202309": rows},
            start="2023-09-01",
            end="2023-09-05",
            root_column="option_root",
        )
        self.assertEqual(tables["daily_summary"].iloc[0]["duplicate_key_rows"], 0)
        self.assertFalse(audit["contract_identity_certified"])
        self.assertFalse(audit["fixings_certified"])
