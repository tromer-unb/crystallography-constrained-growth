from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_growth_cli_help():
    p = subprocess.run(
        [sys.executable, "-m", "ccgrowth.growth", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr
    assert "--site-mode" in p.stdout
    assert "--backend" in p.stdout


def test_si111_advisor_recovers_six_column_motif():
    cif = ROOT / "cases" / "si_111_gpaw" / "Si_111.cif"
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "ccgrowth.parameter_advisor",
            "--structure", str(cif),
            "--template", str(cif),
            "--axis", "z",
            "--species", "Si",
            "--quiet-command",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr
    assert "Structural motif period    : 6" in p.stdout
    assert "2.357" in p.stdout
    assert "0.785" in p.stdout


def test_manuscript_figure_scripts_are_numbered_in_order():
    expected = [f"fig0{i}_" for i in range(1, 8)]
    names = sorted(p.name for p in (ROOT / "figures").glob("fig*.py"))
    assert len(names) == 7
    for prefix, name in zip(expected, names):
        assert name.startswith(prefix)


def _xyz_n(path: Path) -> int:
    return int(path.read_text(errors="replace").splitlines()[0])


def test_si110_checkpoint_is_canonical_final_state():
    d = ROOT / "results" / "silicon" / "110"
    df = pd.read_csv(d / "grafeno_cutoff.csv")
    final_n = int(df["n_atoms"].iloc[-1])
    assert final_n == 109
    assert _xyz_n(d / "grafeno_cutoff_checkpoint.xyz") == final_n
    assert _xyz_n(d / "grafeno_cutoff_final_legacy_mismatch.xyz") == 82


def test_case_commands_reference_existing_inputs():
    import re

    case_root = ROOT / "cases"
    scripts = sorted(case_root.glob("*/run.sh"))
    assert len(scripts) == 20
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "ccgrowth" in text
        refs = re.findall(r"--(?:template-)?cif\s+([^\s\\]+)", text)
        assert refs, f"No CIF input found in {script.relative_to(ROOT)}"
        for ref in refs:
            p = script.parent / ref.strip('"\'')
            assert p.is_file(), f"Missing input {p.relative_to(ROOT)}"


def test_case_manifest_matches_case_directories():
    import csv

    with (ROOT / "cases" / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    documented = {row["case"] for row in rows}
    actual = {p.parent.name for p in (ROOT / "cases").glob("*/run.sh")}
    assert len(rows) == 20
    assert documented == actual
    for row in rows:
        case_dir = ROOT / "cases" / row["case"]
        for name in filter(None, row["input_files"].split(";")):
            assert (case_dir / name).is_file(), f"Missing documented input: {row['case']}/{name}"
