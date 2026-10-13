from fixtures import study_path_fixture, study_research_repository, study_trial_fixture
from fixtures import run_path
from fixtures import gain_report
from dataclasses import replace, asdict
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from paths import HistoricalSettings, FixedContractPaths, funded_summaries
from inference import stationary_weights, GainAccumulator
from paths import METHODS
from study import HistoricalStudy, ArtifactCache, study_lock, frame
from hedging import PastOnlyEmpiricalMV, EmpiricalMVSettings
from fixtures import empirical_fixture
from pricing import BlackFixedIVBenchmark
from ledger import HedgeLedgerSettings


class FixedPathsTests(unittest.TestCase):

    def test_schedules_and_blocks_are_validated(self):
        for kwargs in (
            {"schedules": ((5, 0),)},
            {"schedules": ((2, 3),)},
            {"schedules": ((1, 1), (1, 1))},
            {"block_lengths": (5,)},
            {"bootstrap_replicates": 10},
            {"start": "bad"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HistoricalSettings(**kwargs)

    def test_reference_sessions_not_calendar_days(self):
        p, entry, _, _ = study_path_fixture()
        planner = FixedContractPaths(p)
        self.assertEqual(
            planner.dates(entry["quote_date"], 1), ["2023-09-01", "2023-09-04"]
        )

    def test_daily_rebalance_changes_the_held_hedge(self):
        p, entry, lookup, decisions = study_path_fixture()
        settings = replace(
            HistoricalSettings(),
            funding=dict(lend_rate=0.0, borrow_rate=0.0, hedge_fee_bps=1.0),
        )
        for i, key in enumerate(decisions):
            decisions[key]["black_delta"] = 0.1 * i
        planner = FixedContractPaths(p, settings)
        args = planner.path(entry, 5, 1, lookup, decisions, set(planner.sessions))
        trials, coverage, _ = run_path(*args, settings)
        black = next(
            (r for r in trials if r["strategy"] == "black" and r["scenario"] == "mid")
        )
        self.assertAlmostEqual(
            black["net_pnl"], -2.5 + sum((0.1 * i for i in range(5)))
        )
        self.assertTrue(all((r["status"] == "compared" for r in coverage)))

    def test_five_session_rebalance_holds_entry_delta(self):
        p, entry, lookup, decisions = study_path_fixture()
        settings = replace(
            HistoricalSettings(),
            funding=dict(lend_rate=0.0, borrow_rate=0.0, hedge_fee_bps=1.0),
        )
        for i, key in enumerate(decisions):
            decisions[key]["black_delta"] = 0.1 * i
        planner = FixedContractPaths(p, settings)
        trials, _, _ = run_path(
            *planner.path(entry, 5, 5, lookup, decisions, set(planner.sessions)),
            settings,
        )
        black = next(
            (r for r in trials if r["strategy"] == "black" and r["scenario"] == "mid")
        )
        self.assertAlmostEqual(black["net_pnl"], -2.5)

    def test_missing_midpath_quote_does_not_drop_entry_or_bridge_gap(self):
        p, entry, lookup, decisions = study_path_fixture()
        planner = FixedContractPaths(p)
        del lookup[next((k for k in lookup if k[0] == planner.sessions[2]))]
        meta, state, failed, rows = planner.path(
            entry, 5, 1, lookup, decisions, set(planner.sessions)
        )
        trials, coverage, _ = run_path(meta, state, failed, rows)
        self.assertEqual(trials, [])
        self.assertEqual(len(coverage), len(METHODS) * 3)
        self.assertTrue(
            all(
                (r["status"] == "quote_unavailable_in_research_scope" for r in coverage)
            )
        )
        self.assertEqual(meta["entry_id"], entry["entry_id"])

    def test_excluded_session_cannot_be_skipped(self):
        p, entry, lookup, decisions = study_path_fixture()
        p.loc[2, "session_policy"] = "early_close_clock_review"
        planner = FixedContractPaths(p)
        self.assertEqual(
            planner.path(entry, 5, 1, lookup, decisions, set(planner.sessions))[1],
            "session_excluded",
        )

    def test_prefix_pending_is_distinct_from_sample_end(self):
        p, entry, lookup, decisions = study_path_fixture()
        planner = FixedContractPaths(p)
        self.assertEqual(
            planner.path(entry, 5, 1, lookup, decisions, set(planner.sessions[:2]))[1],
            "pending_run_prefix",
        )
        p = p.iloc[:2]
        planner = FixedContractPaths(p)
        self.assertEqual(
            planner.path(entry, 5, 1, lookup, decisions, set(planner.sessions))[1],
            "sample_end",
        )

    def test_calendar_support_outside_data_is_not_an_endpoint(self):
        p, entry, lookup, decisions = study_path_fixture()
        settings = replace(
            HistoricalSettings(), start=p.quote_date.iloc[0], end=p.quote_date.iloc[1]
        )
        planner = FixedContractPaths(p, settings)
        self.assertEqual(
            planner.path(entry, 5, 1, lookup, decisions, set(planner.sessions))[1],
            "sample_end",
        )

    def test_no_delta_is_needed_at_liquidation(self):
        p, entry, lookup, decisions = study_path_fixture()
        planner = FixedContractPaths(p)
        del decisions[next((k for k in decisions if k[0] == planner.sessions[1]))]
        trials, flags, _ = run_path(
            *planner.path(entry, 1, 1, lookup, decisions, set(planner.sessions))
        )
        self.assertEqual(len(trials), len(METHODS) * 3)

    def test_failed_hedge_has_no_black_fallback(self):
        p, entry, lookup, decisions = study_path_fixture()
        planner = FixedContractPaths(p)
        decisions[next(iter(decisions))]["ah_status"] = "shape_flag"
        trials, flags, _ = run_path(
            *planner.path(entry, 1, 1, lookup, decisions, set(planner.sessions))
        )
        self.assertFalse(any((r["strategy"] == "ah_pde" for r in trials)))
        self.assertTrue(
            all(
                (
                    r["status"] == "shape_flag"
                    for r in flags
                    if r["strategy"] == "ah_pde"
                )
            )
        )
        self.assertTrue(any((r["strategy"] == "black" for r in trials)))

    def test_current_contract_requests_are_unioned_across_entries(self):
        p, entry, _, _ = study_path_fixture()
        planner = FixedContractPaths(p)
        twice = [entry, dict(entry, entry_id="different-entry")]
        requested = planner.requests(planner.sessions[1], twice, pd.DataFrame([entry]))
        self.assertEqual(len(requested), 1)

    def test_fixing_change_is_rejected(self):
        p, entry, lookup, decisions = study_path_fixture()
        lookup[next((k for k in lookup if k[0] == p.quote_date.iloc[1]))][
            "assumed_fixing_utc"
        ] = pd.Timestamp("2023-11-29T21:00:00Z")
        with self.assertRaisesRegex(ValueError, "fixing"):
            FixedContractPaths(p).path(
                entry, 1, 1, lookup, decisions, set(p.quote_date)
            )

    def test_expiry_crossing_does_not_invent_settlement(self):
        p, entry, lookup, decisions = study_path_fixture()
        entry["expire_date"] = p.quote_date.iloc[1]
        p0 = FixedContractPaths(p)
        observation = lookup[next(iter(lookup))]
        lookup[entry["quote_date"], "UNKNOWN", entry["expire_date"], 100.0, "call"] = (
            observation
        )
        self.assertEqual(
            p0.path(entry, 1, 1, lookup, decisions, set(p.quote_date))[1],
            "expiry_before_or_on_path_session",
        )

    def test_vectorized_ledger_matches_scalar_with_asymmetric_funding_and_costs(self):
        p, entry, lookup, decisions = study_path_fixture()
        settings = replace(
            HistoricalSettings(),
            funding=dict(lend_rate=0.01, borrow_rate=0.12, hedge_fee_bps=35.0),
        )
        for i, key in enumerate(decisions):
            for _, delta in METHODS.values():
                if delta:
                    decisions[key][delta] = 1.2 * np.sin(i / 2) - 0.3
        planner = FixedContractPaths(p, settings)
        trials, _, ledgers = run_path(
            *planner.path(entry, 10, 1, lookup, decisions, set(planner.sessions)),
            settings,
        )
        self.assertEqual(len(ledgers), 21)
        self.assertLess(max((r["max_closed_form_error"] for r in trials)), 1e-10)

    def test_vectorized_accounting_rejects_unsupported_unit_conventions(self):
        p, _, lookup, _ = study_path_fixture()
        observed = [lookup[k] for k in list(lookup)[:2]]
        with self.assertRaisesRegex(ValueError, "conventions"):
            funded_summaries(
                observed, np.zeros((1, 2)), HedgeLedgerSettings(option_multiplier=100)
            )

    def test_dst_elapsed_time_is_used_in_funding(self):
        p, entry, lookup, decisions = study_path_fixture(count=2)
        p["quote_date"] = ["2023-11-03", "2023-11-06"]
        entry["quote_date"] = "2023-11-03"
        entry["quote_timestamp_utc"] = pd.Timestamp("2023-11-03T20:00:00Z")
        old = list(lookup.values())
        lookup, mapped = ({}, {})
        decision = next(iter(decisions.values()))
        for i, date in enumerate(p.quote_date):
            row = dict(
                old[i],
                quote_date=date,
                quote_timestamp_utc=pd.Timestamp(
                    date + " 16:00", tz="America/New_York"
                ),
            )
            key = (date, *[row[k] for k in ["root", "expire_date", "strike", "kind"]])
            lookup[key], mapped[key] = (row, decision)
        planner = FixedContractPaths(p)
        trials, _, ledgers = run_path(
            *planner.path(entry, 1, 1, lookup, mapped, set(p.quote_date))
        )
        self.assertAlmostEqual(ledgers[0].interval_years.iloc[1], 73 / (365 * 24))


class UncertaintyTests(unittest.TestCase):

    def test_stationary_resamples_have_one_weight_per_date(self):
        w = stationary_weights(80, 200, 20, np.random.default_rng(7))
        self.assertEqual(w.shape, (200, 80))
        self.assertTrue(np.all(w.sum(axis=1) == 80))
        self.assertTrue((w >= 0).all())

    def test_gain_is_pooled_sse_ratio_without_demeaning(self):
        trials, dates = study_trial_fixture()
        report, _ = gain_report(trials, dates)
        rows = report.loc[report.dimension.eq("overall")]
        self.assertTrue(np.allclose(rows.gain, 0.75))
        self.assertTrue(np.allclose(rows.gain_ci_low, 0.75))
        self.assertTrue(rows.resolved_gain_sign.eq(1).all())

    def test_streaming_statistics_match_nonstreaming_ratios_and_intervals(self):
        trials, dates = study_trial_fixture(100)
        expected, _ = gain_report(trials, dates)
        accumulator = GainAccumulator(dates)
        for _, block in trials.groupby("quote_date"):
            accumulator.add(block)
        actual = accumulator.report()
        keys = [
            "strategy",
            "scenario",
            "holding_sessions",
            "rebalance_sessions",
            "dimension",
            "group",
            "mean_block_sessions",
        ]
        joined = expected.merge(
            actual, on=keys, suffixes=("_e", "_a"), validate="one_to_one"
        )
        for name in (
            "gain",
            "mae_improvement",
            "mse_improvement",
            "gain_ci_low",
            "gain_ci_high",
            "mae_ci_low",
            "mae_ci_high",
        ):
            np.testing.assert_allclose(
                joined[name + "_a"], joined[name + "_e"], equal_nan=True, atol=1e-12
            )

    def test_pairwise_coverage_and_all_method_intersection_differ(self):
        trials, dates = study_trial_fixture(25)
        trials = trials.loc[
            ~((trials.entry_id == dates[0] + "-call") & trials.strategy.eq("lv_smile"))
        ]
        pairwise, _ = gain_report(trials, dates)
        common, _ = gain_report(trials, dates, common_strategies=True)
        a = pairwise.loc[
            pairwise.dimension.eq("overall") & pairwise.strategy.eq("ah_pde")
        ].iloc[0]
        b = common.loc[
            common.dimension.eq("overall") & common.strategy.eq("ah_pde")
        ].iloc[0]
        self.assertEqual(a.comparisons, 50)
        self.assertEqual(b.comparisons, 49)

    def test_zero_black_sse_is_not_infinite_gain(self):
        trials, dates = study_trial_fixture(25)
        trials.loc[trials.strategy.eq("black"), "net_pnl"] = 0.0
        result, _ = gain_report(trials, dates)
        self.assertTrue(result.gain.isna().all())
        self.assertTrue(result.gain_status.eq("zero_black_sse").all())

    def test_short_prefix_does_not_emit_confidence_intervals(self):
        trials, dates = study_trial_fixture(2)
        result, _ = gain_report(trials, dates)
        self.assertTrue(result.ci_status.eq("insufficient_dates").all())

    def test_bootstrap_is_reproducible(self):
        trials, dates = study_trial_fixture(30)
        a, _ = gain_report(trials, dates)
        b, _ = gain_report(trials, dates)
        pd.testing.assert_frame_equal(a, b)

    def test_duplicate_strategy_outcomes_are_rejected(self):
        trials, dates = study_trial_fixture(2)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            gain_report(pd.concat([trials, trials.iloc[:1]]), dates)

    def test_roundoff_differences_do_not_resolve_a_positive_gain(self):
        trials, dates = study_trial_fixture(100)
        trials.loc[~trials.strategy.eq("black"), "net_pnl"] /= 0.5
        trials.loc[~trials.strategy.eq("black"), "net_pnl"] *= 1 - 1e-14
        result, _ = gain_report(trials, dates)
        self.assertTrue(result.resolved_gain_sign.eq(0).all())

    def test_training_endpoint_at_prediction_is_embargoed(self):
        training, _ = empirical_fixture(15)
        current = training.iloc[-6:].copy()
        training["end_timestamp"] = pd.Timestamp(current.quote_timestamp_utc.iloc[0])
        predicted, fits, membership = PastOnlyEmpiricalMV(
            EmpiricalMVSettings(minimum_dates=3, minimum_rows=6)
        ).predict(current, training)
        self.assertTrue(predicted.empirical_mv_status.eq("warmup").all())
        self.assertTrue(membership.empty)

    def test_future_label_values_do_not_change_prediction(self):
        training, _ = empirical_fixture(15)
        current = training.iloc[-6:].copy()
        future = training.iloc[-6:].copy()
        future["entry_id"] = "future-" + future.entry_id
        future["end_timestamp"] = pd.Timestamp(
            current.quote_timestamp_utc.iloc[0]
        ) + pd.Timedelta(days=2)
        estimator = PastOnlyEmpiricalMV(
            EmpiricalMVSettings(minimum_dates=3, minimum_rows=6)
        )
        a = estimator.predict(current, pd.concat([training.iloc[:-6], future]))[0]
        future["end_mid"] = 9999999.0
        b = estimator.predict(current, pd.concat([training.iloc[:-6], future]))[0]
        pd.testing.assert_frame_equal(a, b)


class CacheTests(unittest.TestCase):

    def test_verified_cache_does_not_repeat_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ArtifactCache(tmp)
            calls = []

            def action(folder):
                calls.append(1)
                (folder / "value.txt").write_text("fixed")
                return {"status": "done"}

            cache.get("dated", {"x": 1}, action)
            cache.get("dated", {"x": 1}, action)
            self.assertEqual(len(calls), 1)
            self.assertEqual(cache.timing[-1]["origin"], "verified_cache")

    def test_output_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ArtifactCache(tmp)

            def action(folder):
                (folder / "value.txt").write_text("fixed")
                return {}

            folder, _ = cache.get("dated", {}, action)
            (folder / "value.txt").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Cached output changed"):
                cache.get("dated", {}, action)

    def test_signature_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ArtifactCache(tmp)
            cache.get("dated", {"a": 1}, lambda _: {})
            with self.assertRaisesRegex(ValueError, "identity"):
                cache.get("dated", {"a": 2}, lambda _: {})

    def test_concurrent_invocation_is_rejected_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            with study_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with study_lock(tmp):
                        pass
            self.assertFalse((Path(tmp) / "active.lock").exists())


class IntegratedHistoryTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.repo, cls.config, cls.days = study_research_repository(cls.temp.name)
        cls.study = HistoricalStudy(cls.config, cls.repo)
        cls.study.prepare_calendar()
        cls.prefix = cls.study.run(cls.days[1], progress=lambda _: None)
        cls.prefix_coverage = frame(cls.study.output / "summary/coverage.csv")
        cls.resumed = HistoricalStudy(cls.config, cls.repo)
        cls.final = cls.resumed.run(progress=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_prefix_resume_reuses_model_and_pde_decisions(self):
        reused = [
            row
            for row in self.resumed.cache.timing
            if row["origin"] == "verified_cache"
        ]
        self.assertTrue(
            any((row["job"] == f"dates/{self.days[0]}/decisions" for row in reused))
        )
        self.assertEqual(self.final["status"], "completed")

    def test_real_pipeline_has_all_date_records_and_measured_timing(self):
        coverage = frame(self.study.output / "date_coverage.csv")
        self.assertEqual(set(coverage.date), set(self.days))
        self.assertGreater(self.final["elapsed_wall_seconds"], 0.0)
        self.assertTrue(coverage.fitted_models.gt(0).all())

    def test_pending_paths_become_finished_or_explicit_sample_end(self):
        final = frame(self.study.output / "summary/coverage.csv")
        self.assertTrue(self.prefix_coverage.status.eq("pending_run_prefix").any())
        self.assertFalse(final.status.eq("pending_run_prefix").any())
        self.assertTrue(final.status.eq("sample_end").any())

    def test_accounting_reconciles_with_observed_marks(self):
        results = frame(self.study.output / "summary/strategy_results.csv")
        self.assertLess(results.max_reconciliation.max(), 1e-10)
        self.assertTrue(results.holding_sessions.eq(5).any())

    def test_every_selected_entry_has_all_schedules_and_strategy_coverage(self):
        coverage = frame(self.study.output / "summary/coverage.csv")
        self.assertTrue(
            coverage.groupby("entry_id").size().eq(len(METHODS) * 3 * 3).all()
        )

    def test_model_files_remain_date_specific(self):
        index = json.loads((self.study.output / "run_index.json").read_text())
        self.assertEqual(len(index["dates"]), len(self.days))
        self.assertFalse(index["heldout_claim"])
        first = frame(
            Path(index["dates"][self.days[0]]["decision_folder"]) / "model_manifest.csv"
        )
        self.assertTrue(first.quote_date.eq(self.days[0]).all())

    def test_report_and_representative_ledgers_are_inspectable(self):
        report = frame(self.study.output / "summary/gain_summary.csv")
        self.assertTrue(
            {"maturity_bucket", "moneyness_bucket", "delta_bucket"}
            <= set(report.dimension)
        )
        ledgers = frame(self.study.output / "summary/representative_ledgers.csv")
        self.assertLessEqual(
            len(
                ledgers[
                    [
                        "entry_id",
                        "holding_sessions",
                        "rebalance_sessions",
                        "strategy",
                        "scenario",
                    ]
                ].drop_duplicates()
            ),
            2,
        )

    def test_mutating_frozen_raw_data_prevents_resume(self):
        raw = self.repo / "data/raw/spx_eod_201301.txt"
        original = raw.read_bytes()
        try:
            raw.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "Frozen"):
                HistoricalStudy(self.config, self.repo).freeze()
        finally:
            raw.write_bytes(original)

    def test_resumed_and_uninterrupted_decisions_and_results_match(self):
        original_config = json.loads(self.config.read_text())
        original_config["output"] = "outputs/uninterrupted"
        config = self.repo / "straight.json"
        config.write_text(json.dumps(original_config))
        study = HistoricalStudy(config, self.repo)
        study.prepare_calendar()
        study.run(progress=lambda _: None)
        first = frame(self.study.output / "summary/strategy_results.csv")
        second = frame(study.output / "summary/strategy_results.csv")
        pd.testing.assert_frame_equal(first, second)

    def test_explicit_missing_file_retains_all_reference_date_records(self):
        original_config = json.loads(self.config.read_text())
        original_config["raw_directory"] = "data/unavailable"
        original_config["output"] = "outputs/missing_file"
        config = self.repo / "missing.json"
        config.write_text(json.dumps(original_config))
        study = HistoricalStudy(config, self.repo)
        study.prepare_calendar()
        timing = study.run(progress=lambda _: None)
        dates = frame(study.output / "date_coverage.csv")
        self.assertEqual(len(dates), len(self.days))
        self.assertTrue(dates.prepare_status.eq("raw_file_missing").all())
        self.assertEqual(timing["summary"]["selected_entries"], 0)

    def test_stale_calendar_policy_is_detected_before_freeze(self):
        original = json.loads(self.config.read_text())
        original["output"] = "outputs/stale_calendar"
        path = self.repo / "stale.json"
        path.write_text(json.dumps(original))
        study = HistoricalStudy(path, self.repo)
        study.prepare_calendar()
        policy = self.repo / "policy.csv"
        before = policy.read_bytes()
        try:
            policy.write_bytes(before + b"\n")
            with self.assertRaisesRegex(ValueError, "Calendar support"):
                study.freeze()
        finally:
            policy.write_bytes(before)
