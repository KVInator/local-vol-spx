"""One configurable location for notebook inputs, resolved from a project file."""
from dataclasses import dataclass
import json
import os
from pathlib import Path

from artifacts import sha256
from results import StudyResults


@dataclass(frozen=True)
class NotebookInputs:
    study: Path
    validation_folders: tuple[Path, ...] = ()

    @classmethod
    def load(cls, filename: str | Path | None = None) -> "NotebookInputs":
        if filename is None:
            filename = os.environ.get("SPX_NOTEBOOK_CONFIG")
        if filename is None:
            filename = next(
                (
                    parent / "config/notebooks.json"
                    for parent in (Path.cwd(), *Path.cwd().parents)
                    if (parent / "config/notebooks.json").is_file()
                ),
                None,
            )
        if filename is None:
            raise FileNotFoundError(
                "Launch from the project or set SPX_NOTEBOOK_CONFIG."
            )
        filename = Path(filename).expanduser().resolve()
        config = json.loads(filename.read_text())
        repository = filename.parent.parent

        def resolve(value):
            path = Path(value).expanduser()
            return (path if path.is_absolute() else repository / path).resolve()

        study = resolve(config["study"])
        if (
            config.get("freeze_sha256")
            and sha256(study / "freeze.json") != config["freeze_sha256"]
        ):
            raise ValueError("The configured study does not match the pinned freeze.")
        saved = StudyResults(study)
        if (
            config.get("fingerprint")
            and saved.freeze["fingerprint"] != config["fingerprint"]
        ):
            raise ValueError("The configured study fingerprint differs.")
        return cls(
            study, tuple(resolve(p) for p in config.get("validation_folders", []))
        )

    def results(self) -> StudyResults:
        return StudyResults(self.study)
