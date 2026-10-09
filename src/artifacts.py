from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def signature(value):
    content = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(content.encode()).hexdigest()


@contextmanager
def study_lock(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "active.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 384)
    except FileExistsError as error:
        raise RuntimeError(
            f"Another invocation or interrupted process owns {lock}; check the recorded PID before removing a stale lock."
        ) from error
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {
                    "pid": os.getpid(),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                },
                stream,
            )
        yield
    finally:
        lock.unlink(missing_ok=True)


class ArtifactCache:
    """Run an operation once, then verify its saved outputs before reuse."""

    def __init__(self, parent):
        self.parent = Path(parent)
        self.timing = []

    def get(self, key, identity, action, **arguments):
        folder = self.parent / key
        audit_file = folder / "cache.json"
        started = perf_counter()
        origin = "executed"
        if audit_file.exists():
            audit = json.loads(audit_file.read_text())
            if audit["signature"] != signature(identity):
                raise ValueError(f"Frozen cache identity changed: {key}")
            for name, expected in audit["output_sha256"].items():
                if sha256(folder / name) != expected:
                    raise ValueError(f"Cached output changed: {folder / name}")
            origin = "verified_cache"
        else:
            folder.mkdir(parents=True, exist_ok=True)
            metadata = action(folder, **arguments)
            audit = {
                "signature": signature(identity),
                "identity": identity,
                "metadata": metadata,
                "output_sha256": {
                    str(path.relative_to(folder)): sha256(path)
                    for path in sorted(folder.rglob("*"))
                    if path.is_file()
                    and path.name != "cache.json"
                    and (not path.name.endswith(".tmp"))
                },
                "execution_wall_seconds": perf_counter() - started,
            }
            atomic_json(audit_file, audit)
        self.timing.append(
            {
                "job": key,
                "origin": origin,
                "invocation_wall_seconds": perf_counter() - started,
                "producer_wall_seconds": audit["execution_wall_seconds"],
            }
        )
        return (folder, audit["metadata"])


class PinnedInputs:
    """Track exactly the saved files consumed by an analysis."""

    def __init__(self, index_file, repository=None):
        self.index_file = Path(index_file).resolve()
        self.repository = Path(repository or Path.cwd()).resolve()
        self.inputs = {}

    def read_json(self, path):
        path = Path(path)
        self.inputs[str(path)] = sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def resolve(self, value):
        path = Path(value)
        return (path if path.is_absolute() else self.repository / path).resolve()

    def read_pinned_csv(self, path, expected, dtype, legacy_input_pins=()):
        import pandas as pd

        path = Path(path)
        actual = sha256(path)
        if expected != actual and (
            not (expected is None and actual in legacy_input_pins)
        ):
            raise ValueError(f"Checksum mismatch or missing output pin: {path}")
        self.inputs[str(path)] = actual
        return pd.read_csv(path, dtype=dtype)

    def verify_unchanged(self):
        for path, digest in self.inputs.items():
            if sha256(path) != digest:
                raise ValueError(f"Input changed during analysis: {path}")

    def safe_output(self, output):
        output = Path(output).resolve()
        for folder in (self.index_file.parent, *self.folders.values()):
            if output.is_relative_to(folder) or folder.is_relative_to(output):
                raise ValueError("Use a separate analysis output folder.")
        return output


def require(frame, columns, label, keys=None):
    missing = sorted(set(columns) - set(frame.columns))
    if missing or frame.columns.has_duplicates:
        raise ValueError(f"{label}: missing or duplicate columns: {missing}")
    if keys and (frame[keys].isna().any().any() or frame.duplicated(keys).any()):
        raise ValueError(f"{label}: missing or duplicate keys {keys}")


class CsvTables:
    """Stream study tables to temporary files, then publish completed CSVs."""

    def __init__(self, folder, names):
        self.folder = Path(folder)
        self.temporary = {name: self.folder / (name + ".tmp") for name in names}
        self.columns = {}
        for path in self.temporary.values():
            path.unlink(missing_ok=True)

    def append(self, name, table):
        if table.empty:
            return
        columns = self.columns.setdefault(name, list(table.columns))
        if set(table.columns) != set(columns):
            raise ValueError(f"Outcome CSV schema changed: {name}")
        path = self.temporary[name]
        table[columns].to_csv(path, mode="a", header=not path.exists(), index=False)

    def finish(self):
        import pandas as pd

        for name, path in self.temporary.items():
            if not path.exists():
                pd.DataFrame(
                    columns=["entry_id", "quote_date", "strategy", "scenario", "status"]
                ).to_csv(path, index=False)
            path.replace(self.folder / name)
