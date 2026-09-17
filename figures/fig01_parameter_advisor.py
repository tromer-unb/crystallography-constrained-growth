#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Results Fig. 2 — Automatic crystallographic parameterization and ablation tests

Place this file at:
    PAPER/figure2/figure2_geometric_advisor.py

It reads directly from the current PAPER data tree:

    PAPER/graphene/
    PAPER/graphene/Y/
    PAPER/silicon/111/
    PAPER/graphitte/
    PAPER/SiC/001/

The four panels are:

(a) Graphene X/Y:
    direct measurement of the observed inter-column gap sequences.
    X is nearly uniform; Y alternates between two distances. This shows why
    replacing a periodic gap sequence by one median spacing is not generally
    valid.

(b) Si(111):
    ablation of the structural motif period. A naive p=2 continuation places
    the first future layer too close to the existing crystal, whereas the
    crystallographically complete p=6 motif yields the expected Si-Si distance.

(c) Graphite growth along z:
    detached-layer nucleation. Requiring one pre-existing covalent neighbor
    suppresses the entire new graphene layer; allowing N_min=0 correctly opens
    the first-layer candidate manifold.

(d) Graphene/SiC:
    active crystallographic-region mask. Carbon in the substrate and carbon in
    graphene are chemically identical, so species-only front detection is
    ambiguous. The fractional-z mask isolates the graphene growth phase without
    removing the SiC substrate from the energetic calculation.

Outputs
-------
Figure1_Geometric_Advisor_Ablations.pdf
Figure1_Geometric_Advisor_Ablations.svg
Figure1_Geometric_Advisor_Ablations.png
Figure1_Geometric_Advisor_Ablations_summary.csv

Dependencies
------------
numpy
pandas
matplotlib

ASE is NOT required.
"""

from pathlib import Path
import re
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "results"
CASES = HERE.parent / "cases"

OUT = "Figure1_Geometric_Advisor_Ablations"

GRAPHENE_X = PAPER / "graphene"
GRAPHENE_Y = PAPER / "graphene" / "Y"
SI111 = PAPER / "silicon" / "111"
GRAPHITE = PAPER / "graphitte"
SIC = PAPER / "SiC" / "001"


# =============================================================================
# COLORS
# =============================================================================

DARK = "#17212B"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"
VERY_LIGHT = "#F4F6F8"

BLUE = "#2878B5"
TEAL = "#26958B"
ORANGE = "#D97706"
RED = "#C94F3D"
GREEN = "#4D8B57"
PURPLE = "#7159A6"
GOLD = "#D5A72E"

CARBON = "#454B52"
SILICON = "#A8AFB7"
ACTIVE_C = "#26A69A"


# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.titlesize": 15.5,
    "axes.labelsize": 14,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "axes.linewidth": 1.35,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 4.5,
    "ytick.major.size": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def panel(ax, letter, x=-0.115, y=1.085):
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=23,
        fontweight="bold",
        ha="left",
        va="top",
        color="black",
        clip_on=False,
    )


def note_box(ax, x, y, text, fc="white", ec=LIGHT_GRAY,
             fontsize=9.4, ha="left", va="top"):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight="bold",
        color=DARK,
        bbox=dict(
            boxstyle="round,pad=0.32",
            fc=fc,
            ec=ec,
            lw=1.1,
            alpha=0.94,
        ),
        zorder=20,
    )


# =============================================================================
# EXTENDED XYZ READER
# =============================================================================

def read_extxyz(path):
    """Read symbols, Cartesian positions, cell and PBC from extended XYZ."""

    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ: {path}")

    n = int(lines[0].strip())
    comment = lines[1]

    lattice_match = re.search(r'Lattice="([^"]+)"', comment)

    if lattice_match:
        values = [float(v) for v in lattice_match.group(1).split()]
        if len(values) != 9:
            raise ValueError(f"Invalid Lattice in {path}")
        cell = np.asarray(values, float).reshape(3, 3)
    else:
        cell = np.eye(3)

    pbc_match = re.search(r'pbc="([^"]+)"', comment)
    if pbc_match:
        tokens = pbc_match.group(1).split()
        pbc = np.asarray([t.upper().startswith("T") for t in tokens], bool)
    else:
        pbc = np.zeros(3, dtype=bool)

    symbols = []
    xyz = []

    for line in lines[2:2+n]:
        fields = line.split()
        if len(fields) < 4:
            continue
        symbols.append(fields[0])
        xyz.append([float(fields[1]), float(fields[2]), float(fields[3])])

    if len(xyz) != n:
        raise ValueError(f"XYZ count mismatch in {path}: expected {n}, read {len(xyz)}")

    return (
        np.asarray(symbols, object),
        np.asarray(xyz, float),
        np.asarray(cell, float),
        np.asarray(pbc, bool),
    )


# =============================================================================
# MINIMUM-IMAGE DISTANCES FOR GENERAL CELLS
# =============================================================================

def minimum_image_delta(delta, cell, pbc):
    """Return a minimum-image displacement for an arbitrary 3x3 cell."""

    delta = np.asarray(delta, float)
    cell = np.asarray(cell, float)

    try:
        frac = np.linalg.solve(cell.T, delta)
    except np.linalg.LinAlgError:
        return delta

    frac = np.asarray(frac, float)

    for a in range(3):
        if bool(pbc[a]):
            frac[a] -= np.round(frac[a])

    return frac @ cell


def min_distances_to_structure(points, atoms, cell, pbc):
    """Minimum distance from each point to an existing atomistic structure."""

    values = []

    for p in np.asarray(points, float):
        dmin = np.inf

        for r in np.asarray(atoms, float):
            d = minimum_image_delta(p-r, cell, pbc)
            dmin = min(dmin, float(np.linalg.norm(d)))

        values.append(dmin)

    return np.asarray(values, float)


# =============================================================================
# LOG / SCRIPT PARSERS
# =============================================================================

def parse_log_geometry(log_text):
    """Parse growth axis, motif period, gap period and detected gaps."""

    mg = re.search(r"growth_axis\s*=\s*([xyz])", log_text)
    axis = mg.group(1) if mg else None

    m = re.search(
        r"Template extend-columns:.*?"
        r"periodo_colunas=(\d+)\([^)]*\).*?"
        r"periodo_gaps=(\d+)\([^)]*\).*?"
        r"gaps=\[([^\]]+)\]",
        log_text,
    )

    if m:
        p = int(m.group(1))
        q = int(m.group(2))
        gaps = [float(x.strip()) for x in m.group(3).split(",")]
    else:
        p, q, gaps = None, None, []

    mcut = re.search(r"template_connect_cutoff\s*=\s*([0-9.Ee+-]+)", log_text)
    connect = float(mcut.group(1)) if mcut else None

    mn = re.search(r"min_neighbors\s*=\s*(\d+)", log_text)
    min_neighbors = int(mn.group(1)) if mn else None

    mr = re.search(
        r"Restrição cristalográfica de crescimento:\s*"
        r"(?:.*?Z>=([0-9.Ee+-]+)|desligada)",
        log_text,
    )
    restrict_z = None
    if mr and mr.group(1) is not None:
        restrict_z = float(mr.group(1))

    return {
        "axis": axis,
        "p": p,
        "q": q,
        "gaps": gaps,
        "connect_cutoff": connect,
        "min_neighbors": min_neighbors,
        "restrict_z": restrict_z,
    }


def script_float(path, option, default=None):
    """Read a numeric --option VALUE from a shell script."""

    path = Path(path)
    if not path.exists():
        return default

    text = path.read_text(errors="replace")
    m = re.search(rf"{re.escape(option)}\s+([0-9.Ee+\-]+)", text)
    return float(m.group(1)) if m else default


def script_int(path, option, default=None):
    value = script_float(path, option, default=None)
    if value is None:
        return default
    return int(value)


# =============================================================================
# COLUMN GROUPING
# =============================================================================

AXIS = {"x": 0, "y": 1, "z": 2}


def group_columns(xyz, axis, tol=0.2):
    """Group atoms by Cartesian coordinate along a growth axis."""

    g = AXIS[axis]
    order = np.argsort(xyz[:, g])

    groups = []
    current = []
    ref = None

    for idx in order:
        value = float(xyz[idx, g])

        if ref is None or abs(value-ref) <= tol:
            current.append(int(idx))
            if ref is None:
                ref = value
        else:
            groups.append(current)
            current = [int(idx)]
            ref = value

    if current:
        groups.append(current)

    centers = np.asarray([
        float(np.mean(xyz[gidx, g]))
        for gidx in groups
    ])

    return groups, centers


# =============================================================================
# CIF PARSER FOR THE SiC ACTIVE-REGION PANEL
# =============================================================================

def read_simple_cif(path):
    """
    Minimal CIF reader sufficient for the P1 files in PAPER.

    Returns a DataFrame containing symbol, fractional coordinates and Cartesian
    coordinates, plus the 3x3 cell.
    """

    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()

    values = {}

    wanted = {
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    }

    for line in lines:
        fields = line.split()
        if len(fields) >= 2 and fields[0] in wanted:
            values[fields[0]] = float(fields[1])

    a = values["_cell_length_a"]
    b = values["_cell_length_b"]
    c = values["_cell_length_c"]

    alpha = math.radians(values["_cell_angle_alpha"])
    beta = math.radians(values["_cell_angle_beta"])
    gamma = math.radians(values["_cell_angle_gamma"])

    avec = np.array([a, 0.0, 0.0])
    bvec = np.array([b*np.cos(gamma), b*np.sin(gamma), 0.0])

    cx = c*np.cos(beta)
    cy = c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
    cz2 = max(c*c-cx*cx-cy*cy, 0.0)
    cvec = np.array([cx, cy, np.sqrt(cz2)])

    cell = np.vstack([avec, bvec, cvec])

    # Locate atom-site loop.
    headers = []
    data_start = None

    for i, line in enumerate(lines):
        if line.strip() == "loop_":
            local_headers = []
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("_"):
                local_headers.append(lines[j].strip())
                j += 1

            if "_atom_site_type_symbol" in local_headers:
                headers = local_headers
                data_start = j
                break

    if data_start is None:
        raise ValueError(f"No atom-site loop in {path}")

    idx_sym = headers.index("_atom_site_type_symbol")
    idx_fx = headers.index("_atom_site_fract_x")
    idx_fy = headers.index("_atom_site_fract_y")
    idx_fz = headers.index("_atom_site_fract_z")

    rows = []

    for line in lines[data_start:]:
        stripped = line.strip()

        if not stripped or stripped.startswith("_") or stripped == "loop_":
            if rows and (stripped.startswith("_") or stripped == "loop_"):
                break
            continue

        fields = stripped.split()
        if len(fields) < len(headers):
            if rows:
                break
            continue

        try:
            sym = fields[idx_sym]
            frac = np.array([
                float(fields[idx_fx]),
                float(fields[idx_fy]),
                float(fields[idx_fz]),
            ])
        except Exception:
            if rows:
                break
            continue

        cart = frac @ cell

        rows.append({
            "symbol": sym,
            "fx": frac[0],
            "fy": frac[1],
            "fz": frac[2],
            "x": cart[0],
            "y": cart[1],
            "z": cart[2],
        })

    return pd.DataFrame(rows), cell


# =============================================================================
# FIRST FUTURE LAYER FOR A GIVEN MOTIF PERIOD
# =============================================================================

def future_first_layer_from_period(xyz, axis, groups, p, gap):
    """
    Mimic the first step of extend-columns in the positive growth direction.

    Source column for the first future layer:
        nbase - p

    The future longitudinal coordinate is the final existing layer plus the next
    detected gap.
    """

    g = AXIS[axis]
    nbase = len(groups)

    if p <= 0 or p > nbase:
        raise ValueError(f"Invalid motif period p={p} for {nbase} layers")

    centers = np.asarray([
        np.mean(xyz[idx, g])
        for idx in groups
    ])

    new_h = float(centers[-1] + gap)
    source_group = groups[nbase-p]

    points = xyz[source_group].copy()
    points[:, g] = new_h

    return points, new_h


# =============================================================================
# PANEL A — GRAPHENE X/Y GAP SEQUENCES
# =============================================================================


def plot_graphene_gap_panel(ax, summary):
    """Minimal panel: observed graphene gap sequences only."""
    panel(ax, "a", x=-0.10, y=1.07)

    cases = [
        (GRAPHENE_X, "x", BLUE, r"$x$: $q=1$"),
        (GRAPHENE_Y, "y", TEAL, r"$y$: $q=2$"),
    ]

    y_median = None
    for root, axis, color, label in cases:
        log = (root / "grafeno_cutoff.log").read_text(errors="replace")
        meta = parse_log_geometry(log)
        _, xyz, _, _ = read_extxyz(root / "grafeno_cutoff_initial.xyz")
        _, centers = group_columns(xyz, axis, tol=0.2)
        gaps = np.diff(centers)
        transitions = np.arange(1, len(gaps) + 1)

        ax.plot(
            transitions, gaps,
            marker="o" if axis == "x" else "s",
            ms=5.0, lw=2.0,
            color=color, label=label,
        )

        if axis == "y":
            y_median = float(np.median(gaps))
            ax.axhline(y_median, color=TEAL, lw=1.2, ls="--", alpha=0.75)

        summary.append({
            "case": f"graphene_{axis}",
            "motif_period_p": meta["p"],
            "gap_period_q": meta["q"],
            "median_gap_A": float(np.median(gaps)),
            "detected_gaps_A": ";".join(f"{g:.6f}" for g in meta["gaps"]),
        })

    ax.set_xlabel("Column transition", fontweight="bold")
    ax.set_ylabel(r"Gap, $\Delta h$ (Å)", fontweight="bold")
    ax.set_title("Graphene gap periodicity", fontweight="bold", pad=8)
    ax.legend(frameon=False, loc="upper right")

    if y_median is not None:
        ax.text(
            0.98, 0.06,
            rf"median-only: ${y_median:.3f}$ Å",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9.2, color=GRAY,
        )

    style(ax)


def plot_si111_panel(ax, summary):
    """Minimal p=2 versus p=6 Si(111) motif ablation."""
    panel(ax, "b", x=-0.10, y=1.07)

    log = (SI111 / "grafeno_cutoff.log").read_text(errors="replace")
    meta = parse_log_geometry(log)
    _, xyz, cell, pbc = read_extxyz(SI111 / "grafeno_cutoff_initial.xyz")
    groups, _ = group_columns(xyz, "z", tol=0.2)

    q = int(meta["q"])
    pattern = np.asarray(meta["gaps"], float)
    next_gap = float(pattern[(len(groups) - 1) % q])

    naive_p = 2
    correct_p = int(meta["p"])

    naive_points, _ = future_first_layer_from_period(xyz, "z", groups, naive_p, next_gap)
    correct_points, _ = future_first_layer_from_period(xyz, "z", groups, correct_p, next_gap)

    naive_med = float(np.median(min_distances_to_structure(naive_points, xyz, cell, pbc)))
    correct_med = float(np.median(min_distances_to_structure(correct_points, xyz, cell, pbc)))

    min_dist = script_float(CASES / "si_111_gpaw" / "run.sh", "--min-dist", default=1.7)

    vals = [naive_med, correct_med]
    x = np.arange(2)

    ax.bar(
        x, vals, width=0.58,
        color=["#F1D4CE", "#D8EBD9"],
        edgecolor=[RED, GREEN], linewidth=1.6,
    )
    ax.axhline(min_dist, color=RED, lw=1.25, ls="--")

    for xx, yy in zip(x, vals):
        ax.text(xx, yy + 0.06, f"{yy:.3f} Å", ha="center", va="bottom",
                fontsize=10.5, fontweight="bold")

    ax.text(0.98, min_dist + 0.03, rf"minimum allowed = {min_dist:.1f} Å",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8.9, color=RED)

    ax.set_xticks(x)
    ax.set_xticklabels([r"naive $p=2$", rf"advisor $p={correct_p}$"], fontweight="bold")
    ax.set_ylabel("Nearest Si distance (Å)", fontweight="bold")
    ax.set_title("Si(111) stacking motif", fontweight="bold", pad=8)
    ax.set_ylim(0, max(3.0, correct_med + 0.5))
    style(ax)

    summary.append({
        "case": "Si111_motif_ablation",
        "motif_period_correct": correct_p,
        "motif_period_naive": naive_p,
        "next_gap_A": next_gap,
        "naive_first_layer_median_dmin_A": naive_med,
        "correct_first_layer_median_dmin_A": correct_med,
        "min_dist_A": min_dist,
    })


def plot_graphite_panel(ax, summary):
    """Minimal detached-layer ablation: 0 versus available sites."""
    panel(ax, "c", x=-0.10, y=1.07)

    log = (GRAPHITE / "grafeno_cutoff.log").read_text(errors="replace")
    meta = parse_log_geometry(log)
    df = pd.read_csv(GRAPHITE / "grafeno_cutoff.csv")

    nsites = int(df["n_tested_candidates"].iloc[0])
    first_d = float(df["min_distance_A"].iloc[0])
    connect = script_float(CASES / "graphite" / "run.sh", "--template-connect-cutoff",
                           default=meta["connect_cutoff"])

    vals = [0 if first_d > connect else nsites, nsites]
    x = np.arange(2)

    ax.bar(
        x, vals, width=0.58,
        color=["#F1D4CE", "#D8EBD9"],
        edgecolor=[RED, GREEN], linewidth=1.6,
    )

    for xx, yy in zip(x, vals):
        ax.text(xx, yy + max(0.5, 0.02 * max(nsites, 1)), str(yy),
                ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([r"$N_{\min}=1$", r"$N_{\min}=0$"], fontweight="bold")
    ax.set_ylabel("Admissible sites", fontweight="bold")
    ax.set_title("Graphite layer nucleation", fontweight="bold", pad=8)
    ax.set_ylim(0, nsites * 1.22)

    ax.text(
        0.50, 0.93,
        rf"$\Delta z={meta['gaps'][0]:.3f}$ Å $>$ $r_{{\rm conn}}={connect:.1f}$ Å",
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9.4, color=GRAY,
    )
    style(ax)

    summary.append({
        "case": "graphite_detached_layer",
        "interlayer_gap_A": meta["gaps"][0],
        "first_event_min_distance_A": first_d,
        "connect_cutoff_A": connect,
        "first_layer_candidates": nsites,
        "survivors_if_min_neighbors_1": vals[0],
        "survivors_if_min_neighbors_0": vals[1],
    })


def plot_sic_mask_panel(ax, summary):
    """Minimal active-mask panel: show only carbon, because C-vs-C is the point."""
    panel(ax, "d", x=-0.10, y=1.07)

    log = (SIC / "grafeno_cutoff.log").read_text(errors="replace")
    meta = parse_log_geometry(log)
    cutoff = meta["restrict_z"]
    if cutoff is None:
        raise RuntimeError("SiC/001 log does not contain a Z restriction")

    atoms, _ = read_simple_cif(SIC / "grafeno_cutoff_initial.cif")
    carbon = atoms[atoms["symbol"] == "C"].copy()
    active = carbon[carbon["fz"] >= cutoff].copy()
    substrate = carbon[carbon["fz"] < cutoff].copy()

    ax.scatter(substrate["x"], substrate["fz"], s=18, c="#A7ADB4",
               edgecolors="none", alpha=0.58, label="substrate C")
    ax.scatter(active["x"], active["fz"], s=36, c=ACTIVE_C,
               edgecolors="white", linewidths=0.4, alpha=0.98,
               label="active graphene C")

    ax.axhline(cutoff, color=RED, lw=1.35, ls="--")

    front_all = float(carbon["x"].max())
    front_active = float(active["x"].max())

    ax.axvline(front_all, color=GRAY, lw=1.15, ls=":")
    ax.axvline(front_active, color=ACTIVE_C, lw=1.3, ls=":")

    ymax = ax.get_ylim()[1]
    ax.text(front_active, ymax, "masked front", rotation=90,
            ha="right", va="top", fontsize=8.8, color=ACTIVE_C, fontweight="bold")
    ax.text(front_all, ymax, "species-only front", rotation=90,
            ha="right", va="top", fontsize=8.8, color=GRAY, fontweight="bold")

    ax.text(0.98, cutoff + 0.008, rf"$Z_{{\rm frac}}\geq {cutoff:.3f}$",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=8.8, color=RED)

    ax.set_xlabel(r"Growth coordinate, $x$ (Å)", fontweight="bold")
    ax.set_ylabel(r"Fractional $Z$", fontweight="bold")
    ax.set_title("Graphene/SiC active mask", fontweight="bold", pad=8)
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=8.8)
    style(ax)

    summary.append({
        "case": "graphene_SiC_active_mask",
        "restrict_Z_fractional": cutoff,
        "n_total_C": len(carbon),
        "n_active_C": len(active),
        "species_only_front_x_A": front_all,
        "masked_graphene_front_x_A": front_active,
        "front_error_without_mask_A": front_all - front_active,
    })


def main():
    required = [GRAPHENE_X, GRAPHENE_Y, SI111, GRAPHITE, SIC]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing PAPER directories:\n" + "\n".join(map(str, missing)))

    summary = []

    fig, axs = plt.subplots(2, 2, figsize=(13.8, 9.4), facecolor="white")
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.09, top=0.95,
                        wspace=0.27, hspace=0.34)

    plot_graphene_gap_panel(axs[0, 0], summary)
    plot_si111_panel(axs[0, 1], summary)
    plot_graphite_panel(axs[1, 0], summary)
    plot_sic_mask_panel(axs[1, 1], summary)

    out = HERE / OUT
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.06)
    pd.DataFrame(summary).to_csv(HERE / f"{OUT}_summary.csv", index=False)

    print("Generated:")
    for ext in ("pdf", "svg", "png"):
        print(" ", out.with_suffix("." + ext))
    print(" ", HERE / f"{OUT}_summary.csv")

    plt.show()


if __name__ == "__main__":
    main()
