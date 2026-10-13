"""Regenerate curated figures/tables or render the standalone report."""
import argparse
from pathlib import Path

from notebook_config import NotebookInputs
from publication import PublicationArtifacts, ReportRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if not args.figures and not args.report:
        parser.error("Select --figures, --report, or both.")
    repository = Path(__file__).resolve().parents[1]
    if args.figures:
        inputs = NotebookInputs.load(repository / "config/notebooks.json")
        PublicationArtifacts(inputs.results(), repository).export(repository / "docs")
        print("Curated publication figures and tables regenerated.")
    if args.report:
        print("Report:", ReportRenderer(repository).render())


if __name__ == "__main__":
    main()
