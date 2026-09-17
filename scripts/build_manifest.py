#!/usr/bin/env python3
"""Build or check the deterministic checksum manifest for cases/ and results/."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
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


def role_for(rel: Path) -> str:
    if rel.parts[0] == "cases":
        return "case-command" if rel.name == "run.sh" else "case-input"
    return "publication-result"


def render_manifest() -> str:
    files = sorted(
        p
        for archive_root in ARCHIVE_ROOTS
        for p in archive_root.rglob("*")
        if p.is_file()
    )
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(["path", "size_bytes", "sha256", "role"])
    for p in files:
        rel = p.relative_to(ROOT)
        writer.writerow([rel.as_posix(), p.stat().st_size, sha256(p), role_for(rel)])
    return out.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if the committed manifest is stale")
    args = parser.parse_args()
    content = render_manifest()
    if args.check:
        with MANIFEST.open("r", encoding="utf-8", newline="") as f:
            current = f.read()
        if current != content:
            print("reproducibility/manifest.csv is stale; run scripts/build_manifest.py", file=sys.stderr)
            return 1
        print("Manifest is deterministic and up to date.")
        return 0
    with MANIFEST.open("w", encoding="utf-8", newline="") as f:
        f.write(content)
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
