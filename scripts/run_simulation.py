"""Run the controlled Black-Scholes and CEV hedging experiment."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from artifacts import atomic_json
from simulation import KnownModelExperiment, KnownModelSettings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/simulation.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    settings = KnownModelSettings.from_json(args.config)
    if args.plan:
        print(settings.plan())
        return
    stamp = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("outputs/simulation") / stamp
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    KnownModelExperiment(settings).run(output)
    atomic_json(
        output / "timing.json", {"elapsed_wall_seconds": perf_counter() - started}
    )
    print("Saved simulation:", output.resolve())


if __name__ == "__main__":
    main()
