#!/usr/bin/env python3
"""Validate the curated publication archive against its checksum manifest."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "reproducibility" / "manifest.csv"
ARCHIVE_ROOTS = (ROOT / "cases", ROOT / "results")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    missing: list[str] = []
    corrupt: list[tuple[str, str, str]] = []
    wrong_size: list[tuple[str, int, int]] = []
    wrong_role: list[tuple[str, str, str]] = []
    unsafe: list[str] = []

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    paths = [row["path"] for row in rows]
    duplicates = sorted({p for p in paths if paths.count(p) > 1})
    listed = set(paths)

    for row in rows:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            unsafe.append(row["path"])
            continue
        p = ROOT / rel
        if not p.exists():
            missing.append(row["path"])
            continue
        expected_size = int(row["size_bytes"])
        actual_size = p.stat().st_size
        if actual_size != expected_size:
            wrong_size.append((row["path"], expected_size, actual_size))

        expected_role = (
            "case-command"
            if rel.parts[0] == "cases" and rel.name == "run.sh"
            else "case-input"
            if rel.parts[0] == "cases"
            else "publication-result"
        )
        if row["role"] != expected_role:
            wrong_role.append((row["path"], expected_role, row["role"]))

        digest = sha256(p)
        if digest != row["sha256"]:
            corrupt.append((row["path"], row["sha256"], digest))

    actual = {
        p.relative_to(ROOT).as_posix()
        for archive_root in ARCHIVE_ROOTS
        for p in archive_root.rglob("*")
        if p.is_file()
    }
    unlisted = sorted(actual - listed)
    stale = sorted(listed - actual)

    if duplicates or unsafe or missing or wrong_size or wrong_role or corrupt or unlisted or stale:
        for p in duplicates:
            print(f"DUPLICATE MANIFEST ENTRY {p}", file=sys.stderr)
        for p in unsafe:
            print(f"UNSAFE MANIFEST PATH {p}", file=sys.stderr)
        for p in missing:
            print(f"MISSING {p}", file=sys.stderr)
        for p, expected, got in wrong_size:
            print(f"SIZE MISMATCH {p}: expected {expected}, got {got}", file=sys.stderr)
        for p, expected, got in wrong_role:
            print(f"ROLE MISMATCH {p}: expected {expected}, got {got}", file=sys.stderr)
        for p, expected, got in corrupt:
            print(f"HASH MISMATCH {p}\n  expected {expected}\n  got      {got}", file=sys.stderr)
        for p in unlisted:
            print(f"UNLISTED ARCHIVE FILE {p}", file=sys.stderr)
        for p in stale:
            print(f"STALE MANIFEST ENTRY {p}", file=sys.stderr)
        return 1

    print(
        f"Archive validation OK: {len(rows)} files; "
        "coverage, roles, sizes, and SHA-256 checksums all match."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
