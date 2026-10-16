"""Create and compare local file inventories without following symlinks."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Entry:
    """Describe a file by content digest and byte length."""

    size: int
    sha256: str


@dataclass(frozen=True)
class Difference:
    """Separate paths added, removed, or changed between two inventories."""

    added: tuple[str, ...]
    removed: tuple[str, ...]
    modified: tuple[str, ...]


def digest_file(path: Path) -> Entry:
    """Hash one file in bounded chunks and count the bytes actually read."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(64 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return Entry(size, digest.hexdigest())


def scan(root: Path) -> dict[str, Entry]:
    """Build a relative-path inventory, skipping symlinks and special files."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Root must be a real directory")
    inventory = {}

    def report_error(error: OSError) -> None:
        raise error

    for directory, subdirectories, files in os.walk(
        root, followlinks=False, onerror=report_error
    ):
        base = Path(directory)
        subdirectories[:] = sorted(
            name for name in subdirectories if not (base / name).is_symlink()
        )
        for name in sorted(files):
            path = base / name
            if path.is_symlink() or not path.is_file():
                continue
            inventory[path.relative_to(root).as_posix()] = digest_file(path)
    return inventory


def save_inventory(inventory: dict[str, Entry], destination: Path) -> None:
    """Atomically replace a JSON snapshot using a sibling temporary file."""
    payload = {
        "version": 1,
        "files": {key: asdict(value) for key, value in sorted(inventory.items())},
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=".inventory-",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_inventory(source: Path) -> dict[str, Entry]:
    """Read a versioned snapshot and reject unsafe or malformed records."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported inventory format")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError("Expected file records")
    result = {}
    for name, value in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Unsafe relative path")
        if not isinstance(value, dict):
            raise ValueError("Malformed file record")
        size, digest = value.get("size"), value.get("sha256")
        if type(size) is not int or size < 0:
            raise ValueError("Invalid size")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid SHA-256 digest")
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("Invalid SHA-256 digest")
        result[name] = Entry(size, digest)
    return result


def compare(before: dict[str, Entry], after: dict[str, Entry]) -> Difference:
    """Compare inventories using content and size rather than timestamps."""
    return Difference(
        tuple(sorted(after.keys() - before.keys())),
        tuple(sorted(before.keys() - after.keys())),
        tuple(
            sorted(
                name
                for name in before.keys() & after.keys()
                if before[name] != after[name]
            )
        ),
    )


def main() -> None:
    """Run a temporary demo or scan a directory into a JSON snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.root:
        if args.output is None:
            parser.error("--root requires --output outside the scanned directory")
        root = args.root.resolve()
        if args.output.resolve().is_relative_to(root):
            parser.error("Store snapshots outside the scanned directory")
        inventory = scan(root)
        save_inventory(inventory, args.output)
        LOGGER.info("Saved %d files", len(inventory))
    else:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notes.txt").write_text("first", encoding="utf-8")
            before = scan(root)
            (root / "notes.txt").write_text("revised", encoding="utf-8")
            LOGGER.info("Modified: %s", ", ".join(compare(before, scan(root)).modified))


if __name__ == "__main__":
    main()
