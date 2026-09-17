#!/usr/bin/env python3
"""Rebuild manuscript figures and optionally regression-check summary tables."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
REF = ROOT / "results" / "figure_summaries"

JOBS = [
    ("fig01_parameter_advisor.py", "Figure1_Geometric_Advisor_Ablations_summary.csv"),
    ("fig02_si_gpaw_facets.py", "Figure2_Si_GPAW_results_summary.csv"),
    ("fig03_carbon_architectures.py", "Figure3_Structural_Generality_summary.csv"),
    ("fig04_heterointerfaces.py", "Figure4_Heterostructure_Multicomponent_summary.csv"),
    ("fig05_defects.py", "Figure5_Programmable_Defects_summary.csv"),
    ("fig06_backend_portability.py", "Figure6_Backend_Portability_summary.csv"),
    ("fig07_template_free.py", "Figure7_TemplateFree_Graphene_Fe_B_summary.csv"),
]


def compare_tables(got: Path, ref: Path) -> list[str]:
    a = pd.read_csv(got)
    b = pd.read_csv(ref)
    problems: list[str] = []
    if list(a.columns) != list(b.columns):
        problems.append(f"column mismatch: {list(a.columns)} != {list(b.columns)}")
        return problems
    if len(a) != len(b):
        problems.append(f"row-count mismatch: {len(a)} != {len(b)}")
        return problems

    # Absolute repository paths are expected to differ after restructuring.
    path_metadata = {"directory"}
    for col in a.columns:
        if col in path_metadata:
            continue
        sa, sb = a[col], b[col]
        if pd.api.types.is_numeric_dtype(sa) and pd.api.types.is_numeric_dtype(sb):
            xa = pd.to_numeric(sa, errors="coerce").to_numpy(float)
            xb = pd.to_numeric(sb, errors="coerce").to_numpy(float)
            if not np.allclose(xa, xb, rtol=1e-12, atol=1e-12, equal_nan=True):
                problems.append(f"numeric mismatch in {col}")
        else:
            va = sa.fillna("").astype(str).tolist()
            vb = sb.fillna("").astype(str).tolist()
            if va != vb:
                problems.append(f"text mismatch in {col}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help="compare generated summary CSVs against publication references",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    failures: list[str] = []

    for script, summary in JOBS:
        path = FIG / script
        print(f"[figure] {script}", flush=True)
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        if proc.returncode:
            failures.append(f"{script}: exit {proc.returncode}\n{proc.stderr}")
            continue

        out = FIG / summary
        if not out.exists():
            failures.append(f"{script}: missing generated summary {out.name}")
            continue

        if args.check_reference:
            ref = REF / summary
            if not ref.exists():
                failures.append(f"{script}: missing reference {ref}")
                continue
            problems = compare_tables(out, ref)
            if problems:
                failures.append(f"{script}: " + "; ".join(problems))
            else:
                print(f"  regression: MATCH ({summary})")

    if failures:
        print("\nFigure reproduction failures:", file=sys.stderr)
        for f in failures:
            print(f"- {f}", file=sys.stderr)
        return 1

    print("\nAll manuscript figures reproduced successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
