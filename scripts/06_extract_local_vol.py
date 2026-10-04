"""Extract local volatility from an existing calibrated surface."""

import argparse
from pathlib import Path

from local_vol_outputs import extract_local_volatility


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True, type=Path)
    args = parser.parse_args()

    extract_local_volatility(args.surface)


if __name__ == "__main__":
    main()