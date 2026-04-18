from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lv_project.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, OUTPUT_TABLES_DIR
from lv_project.preprocessing import (
    build_clean_call_dataset,
    load_raw_option_directory,
    make_daily_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse raw SPX EOD chain files into a clean call-IV dataset.")
    parser.add_argument("--raw-dir", type=str, default=str(DATA_RAW_DIR))
    parser.add_argument("--pattern", type=str, default="spx_eod_*.txt")
    parser.add_argument("--flat-rate", type=float, default=0.0)
    parser.add_argument("--flat-dividend-yield", type=float, default=0.0)
    parser.add_argument("--min-dte", type=int, default=7)
    parser.add_argument("--max-dte", type=int, default=365)
    parser.add_argument("--min-bid", type=float, default=0.05)
    parser.add_argument("--max-spread-pct-mid", type=float, default=0.35)
    args = parser.parse_args()

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_option_directory(raw_dir=args.raw_dir, pattern=args.pattern)
    clean_calls = build_clean_call_dataset(
        raw_df=raw,
        flat_rate=args.flat_rate,
        flat_dividend_yield=args.flat_dividend_yield,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        min_bid=args.min_bid,
        max_spread_pct_mid=args.max_spread_pct_mid,
    )
    daily_summary = make_daily_summary(clean_calls)

    clean_path = DATA_PROCESSED_DIR / "spx_calls_clean.csv"
    summary_path = OUTPUT_TABLES_DIR / "spx_calls_daily_summary.csv"

    clean_calls.to_csv(clean_path, index=False)
    daily_summary.to_csv(summary_path, index=False)

    print("Prepared clean call dataset")
    print(f"raw rows                  : {len(raw):,}")
    print(f"clean call rows           : {len(clean_calls):,}")
    print(f"distinct quote dates      : {clean_calls['quote_date'].nunique():,}")
    print(f"distinct expiries         : {clean_calls['expiry'].nunique():,}")
    print(f"date range                : {clean_calls['quote_date'].min().date()} -> {clean_calls['quote_date'].max().date()}")
    print(f"clean dataset saved to    : {clean_path}")
    print(f"daily summary saved to    : {summary_path}")


if __name__ == "__main__":
    main()