"""Export figures and a short report from saved historical results."""

import argparse
from pathlib import Path

from results import StudyResults


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print("Report:", StudyResults(args.study).export(args.output))


if __name__ == "__main__":
    main()
