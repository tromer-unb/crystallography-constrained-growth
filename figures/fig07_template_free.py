#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 7 — Template-free interaction of finite bilayer graphene with B and Fe

Place this file at:
    PAPER/figure7/figure7.py

Reads EXACTLY from:
    PAPER/graphitte_Fe_B/

Expected files:
    graphite_FeB_lateral.csv
    graphite_FeB_lateral.log
    graphite_FeB_lateral_initial.xyz
    graphite_FeB_lateral_final.xyz
    graphite_FeB_lateral_checkpoint.xyz

The figure is intentionally compact:

(a) Real final structure, top view (x-y)
(b) Real final structure, side view (x-z), revealing the bilayer geometry
(c) Accepted incorporation sequence for C, B and Fe
(d) Species-resolved unrelaxed insertion energies

Initial atoms are separated from atoms incorporated during the simulation
using n_initial from the CSV. The structural panels are rendered directly
from the final XYZ coordinates; they are not schematic reconstructions.

Outputs
-------
Figure7_TemplateFree_Graphene_Fe_B.pdf
Figure7_TemplateFree_Graphene_Fe_B.svg
Figure7_TemplateFree_Graphene_Fe_B.png
Figure7_TemplateFree_Graphene_Fe_B_summary.csv

Dependencies
------------
numpy
pandas
matplotlib

ASE is NOT required.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "results"
DATA = PAPER / "graphitte_Fe_B"

CSV_FILE = DATA / "graphite_FeB_lateral.csv"
LOG_FILE = DATA / "graphite_FeB_lateral.log"

INITIAL_XYZ = DATA / "graphite_FeB_lateral_initial.xyz"
FINAL_XYZ = DATA / "graphite_FeB_lateral_final.xyz"
CHECKPOINT_XYZ = DATA / "graphite_FeB_lateral_checkpoint.xyz"

OUT = "Figure7_TemplateFree_Graphene_Fe_B"


# =============================================================================
# COLORS / STYLE
# =============================================================================

DARK = "#17212B"
GRAY = "#6B7280"
LIGHT_GRAY = "#B9BEC5"
WHITE = "#FFFFFF"

# Initial bilayer graphene
C_INITIAL = "#9DA3AA"

# Carbon incorporated during the template-free run
C_ADDED = "#3B78A8"

# Impurities
B_COLOR = "#D98922"
FE_COLOR = "#C94F3D"

# Bonds
BOND_CC = "#AEB4BA"
BOND_CB = "#C79A54"
BOND_CFE = "#A66A60"
BOND_OTHER = "#A7ADB4"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.0,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.5,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 9.2,
    "axes.linewidth": 1.25,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
    "xtick.major.size": 4.2,
    "ytick.major.size": 4.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def panel(ax, letter, x=-0.12, y=1.08):
    ax.text(
        x, y, letter,
        transform=ax.transAxes,
        fontsize=22,
        fontweight="bold",
        ha="left",
        va="top",
        color="black",
        clip_on=False,
    )


# =============================================================================
# XYZ / CSV HELPERS
# =============================================================================

def read_xyz(path):
    """
    Read a standard or ASE extended XYZ without requiring ASE.
    """

    path = Path(path)

    lines = path.read_text(
        errors="replace"
    ).splitlines()

    if len(lines) < 2:
        raise ValueError(
            f"Invalid XYZ file: {path}"
        )

    n = int(
        lines[0].strip()
    )

    symbols = []
    xyz = []

    for line in lines[2:2+n]:

        fields = line.split()

        if len(fields) < 4:
            continue

        symbols.append(
            fields[0]
        )

        xyz.append([
            float(fields[1]),
            float(fields[2]),
            float(fields[3]),
        ])

    if len(xyz) != n:

        raise ValueError(
            f"XYZ mismatch in {path}: "
            f"expected {n}, read {len(xyz)}"
        )

    return (
        np.asarray(symbols, object),
        np.asarray(xyz, float),
    )


def xyz_n(path):

    path = Path(path)

    if not path.exists():
        return None

    try:

        return int(
            path.read_text(
                errors="replace"
            ).splitlines()[0].strip()
        )

    except Exception:

        return None


def choose_final_structure(
    expected_n,
):
    """
    Prefer final.xyz when it matches the CSV. Otherwise use checkpoint.xyz.
    """

    candidates = [
        (FINAL_XYZ, "final.xyz"),
        (CHECKPOINT_XYZ, "checkpoint.xyz"),
    ]

    for path, source in candidates:

        if (
            path.exists()
            and
            xyz_n(path) == expected_n
        ):

            return (
                path,
                source,
            )

    existing = [

        (
            path,
            source,
            xyz_n(path),
        )

        for path, source
        in candidates

        if (
            path.exists()
            and
            xyz_n(path) is not None
        )
    ]

    if not existing:

        raise FileNotFoundError(
            "Neither final nor checkpoint XYZ was found."
        )

    path, source, n = min(

        existing,

        key=lambda item:
        abs(
            item[2] - expected_n
        ),
    )

    warnings.warn(

        f"No XYZ has the final CSV atom count N={expected_n}. "
        f"Using {source}, N={n}. "
        "Check run provenance before publication."

    )

    return (
        path,
        source + " [COUNT MISMATCH]",
    )


def find_column(
    df,
    exact=(),
    contains=(),
):
    """
    Robustly locate a CSV column.
    """

    for name in exact:

        if name in df.columns:
            return name

    low = {

        str(c).lower():
        c

        for c
        in df.columns
    }

    for token in contains:

        token = token.lower()

        for lc, original in low.items():

            if token in lc:
                return original

    return None


def species_column(df):

    return find_column(
        df,
        exact=(
            "deposited_species",
            "species",
            "intended_species",
        ),
        contains=(
            "deposited_species",
            "species",
        ),
    )


# =============================================================================
# STRUCTURAL ANALYSIS
# =============================================================================

def bond_cutoff(
    species_i,
    species_j,
):
    """
    Visualization cutoffs only.

    They are used to render the actual final coordinates;
    they do not enter the growth simulation or any energetic calculation.
    """

    a = str(species_i)
    b = str(species_j)

    pair = frozenset(
        (
            a,
            b,
        )
    )

    if (
        a == "C"
        and
        b == "C"
    ):
        return 1.85

    if pair == frozenset(("B", "C")):
        return 2.00

    if pair == frozenset(("Fe", "C")):
        return 2.45

    if pair == frozenset(("B", "Fe")):
        return 2.55

    if (
        a == "B"
        and
        b == "B"
    ):
        return 2.00

    if (
        a == "Fe"
        and
        b == "Fe"
    ):
        return 2.85

    return None


def bond_color(
    species_i,
    species_j,
):

    a = str(species_i)
    b = str(species_j)

    pair = frozenset(
        (
            a,
            b,
        )
    )

    if (
        a == "C"
        and
        b == "C"
    ):
        return BOND_CC

    if pair == frozenset(("B", "C")):
        return BOND_CB

    if pair == frozenset(("Fe", "C")):
        return BOND_CFE

    return BOND_OTHER


def build_bonds(
    symbols,
    xyz,
):
    """
    Infer bonds from the actual final atomic coordinates.
    """

    symbols = np.asarray(
        symbols,
        object,
    )

    xyz = np.asarray(
        xyz,
        float,
    )

    bonds = []

    n = len(
        xyz
    )

    for i in range(
        n - 1
    ):

        delta = (
            xyz[
                i+1:
            ]
            -
            xyz[
                i
            ]
        )

        d = np.linalg.norm(
            delta,
            axis=1,
        )

        for j0, distance in enumerate(
            d
        ):

            j = (
                i
                +
                1
                +
                j0
            )

            cutoff = bond_cutoff(
                symbols[i],
                symbols[j],
            )

            if cutoff is None:
                continue

            if (
                distance > 1e-8
                and
                distance <= cutoff
            ):

                bonds.append(
                    (
                        i,
                        j,
                    )
                )

    return bonds


def two_layer_centers(
    xyz,
    symbols,
    n_initial,
):
    """
    Estimate the two initial graphene-layer z positions using simple
    1D two-means clustering of initial carbon atoms.

    This is used only to place unobtrusive horizontal guides in panel (b).
    """

    sym = np.asarray(
        symbols[
            :n_initial
        ]
    )

    coords = np.asarray(
        xyz[
            :n_initial
        ],
        float,
    )

    z = (
        coords[
            sym == "C",
            2
        ]
    )

    if len(z) < 2:

        return []

    c1 = float(
        np.percentile(
            z,
            25,
        )
    )

    c2 = float(
        np.percentile(
            z,
            75,
        )
    )

    for _ in range(
        30
    ):

        d1 = np.abs(
            z - c1
        )

        d2 = np.abs(
            z - c2
        )

        g1 = (
            z[
                d1 <= d2
            ]
        )

        g2 = (
            z[
                d2 < d1
            ]
        )

        if (
            len(g1) == 0
            or
            len(g2) == 0
        ):
            break

        n1 = float(
            np.mean(
                g1
            )
        )

        n2 = float(
            np.mean(
                g2
            )
        )

        if (
            abs(n1-c1)
            +
            abs(n2-c2)
            <
            1e-8
        ):
            break

        c1 = n1
        c2 = n2

    return sorted(
        [
            c1,
            c2,
        ]
    )


# =============================================================================
# STRUCTURAL RENDERING
# =============================================================================

def atom_style(
    symbol,
    is_initial,
):
    """
    Return color, size, zorder.
    """

    symbol = str(
        symbol
    )

    if symbol == "Fe":

        return (
            FE_COLOR,
            52,
            8,
        )

    if symbol == "B":

        return (
            B_COLOR,
            42,
            7,
        )

    if symbol == "C":

        if is_initial:

            return (
                C_INITIAL,
                21,
                4,
            )

        return (
            C_ADDED,
            27,
            5,
        )

    return (
        GRAY,
        24,
        3,
    )


def draw_structure(
    ax,
    symbols,
    xyz,
    n_initial,
    projection="xy",
    bonds=None,
):
    """
    Render actual simulation coordinates.

    projection:
        'xy' = top view
        'xz' = side view
    """

    symbols = np.asarray(
        symbols,
        object,
    )

    xyz = np.asarray(
        xyz,
        float,
    )

    if bonds is None:

        bonds = build_bonds(
            symbols,
            xyz,
        )

    if projection == "xy":

        a0 = 0
        a1 = 1

    elif projection == "xz":

        a0 = 0
        a1 = 2

    else:

        raise ValueError(
            "projection must be xy or xz"
        )

    # -------------------------------------------------------------------------
    # Real bonds
    # -------------------------------------------------------------------------

    for i, j in bonds:

        ax.plot(

            [
                xyz[
                    i,
                    a0
                ],
                xyz[
                    j,
                    a0
                ],
            ],

            [
                xyz[
                    i,
                    a1
                ],
                xyz[
                    j,
                    a1
                ],
            ],

            color=bond_color(
                symbols[i],
                symbols[j],
            ),

            lw=0.70,

            alpha=0.58,

            zorder=1,
        )

    # -------------------------------------------------------------------------
    # Actual atoms
    # -------------------------------------------------------------------------

    # Plot initial carbon first, then added C, then impurities.
    order = []

    for i, s in enumerate(
        symbols
    ):

        initial = (
            i < n_initial
        )

        if (
            s == "C"
            and
            initial
        ):

            rank = 0

        elif s == "C":

            rank = 1

        elif s == "B":

            rank = 2

        elif s == "Fe":

            rank = 3

        else:

            rank = 1

        order.append(
            (
                rank,
                i,
            )
        )

    for _, i in sorted(
        order
    ):

        initial = (
            i < n_initial
        )

        color, size, zorder = atom_style(
            symbols[i],
            initial,
        )

        ax.scatter(

            [
                xyz[
                    i,
                    a0
                ]
            ],

            [
                xyz[
                    i,
                    a1
                ]
            ],

            s=size,

            c=color,

            edgecolors=(
                "none"
                if (
                    symbols[i] == "C"
                    and
                    initial
                )
                else
                WHITE
            ),

            linewidths=(
                0.0
                if (
                    symbols[i] == "C"
                    and
                    initial
                )
                else
                0.42
            ),

            alpha=(
                0.72
                if (
                    symbols[i] == "C"
                    and
                    initial
                )
                else
                0.98
            ),

            zorder=zorder,
        )

    ax.set_aspect(
        "equal"
        if projection == "xy"
        else
        "auto",
        adjustable="box",
    )

    style(
        ax
    )


# =============================================================================
# PANEL (c): EVENT SEQUENCE
# =============================================================================

def plot_event_sequence(
    ax,
    df,
):

    panel(
        ax,
        "c",
    )

    scol = species_column(
        df
    )

    if scol is None:

        raise RuntimeError(
            "Could not locate deposited species column in CSV."
        )

    step_col = find_column(
        df,
        exact=(
            "step",
        ),
    )

    if step_col is None:

        step = np.arange(
            1,
            len(df)+1,
        )

    else:

        step = pd.to_numeric(
            df[
                step_col
            ],
            errors="coerce",
        ).to_numpy(
            float
        )

    species = (

        df[
            scol
        ]

        .astype(
            str
        )

        .str
        .strip()
    )

    mapping = {
        "C": 0,
        "B": 1,
        "Fe": 2,
    }

    colors = {
        "C": C_ADDED,
        "B": B_COLOR,
        "Fe": FE_COLOR,
    }

    for sp in (
        "C",
        "B",
        "Fe",
    ):

        mask = (
            species
            .str
            .lower()
            ==
            sp.lower()
        )

        if not np.any(
            mask
        ):
            continue

        ax.scatter(

            step[
                mask
            ],

            np.full(
                int(
                    mask.sum()
                ),
                mapping[
                    sp
                ],
                float,
            ),

            s=(
                17
                if sp == "C"
                else
                28
            ),

            c=colors[
                sp
            ],

            edgecolors=(
                "none"
                if sp == "C"
                else
                WHITE
            ),

            linewidths=0.35,

            alpha=0.88,

            label=sp,

            zorder=3,
        )

    ax.set_yticks(
        [
            0,
            1,
            2,
        ]
    )

    ax.set_yticklabels(
        [
            "C",
            "B",
            "Fe",
        ],
        fontweight="bold",
    )

    ax.set_ylim(
        -0.55,
        2.55,
    )

    ax.set_xlabel(
        "Accepted growth event",
        fontweight="bold",
    )

    ax.set_ylabel(
        "Species",
        fontweight="bold",
    )

    ax.set_title(
        "Template-free incorporation sequence",
        fontweight="bold",
        pad=9,
    )

    style(
        ax
    )


# =============================================================================
# PANEL (d): INSERTION ENERGIES
# =============================================================================

def plot_energy_by_species(
    ax,
    df,
):

    panel(
        ax,
        "d",
    )

    scol = species_column(
        df
    )

    ecol = find_column(
        df,
        exact=(
            "e_ads_unrelaxed_eV",
            "e_ads_unrelaxed",
        ),
        contains=(
            "e_ads_unrelaxed",
            "eads_unrelaxed",
        ),
    )

    if scol is None:

        raise RuntimeError(
            "Could not locate deposited species column."
        )

    if ecol is None:

        raise RuntimeError(
            "Could not locate unrelaxed insertion-energy column."
        )

    species = (

        df[
            scol
        ]

        .astype(
            str
        )

        .str
        .strip()
    )

    energy = pd.to_numeric(
        df[
            ecol
        ],
        errors="coerce",
    )

    names = []
    values = []
    colors = []

    for sp, color in (
        ("C", C_ADDED),
        ("B", B_COLOR),
        ("Fe", FE_COLOR),
    ):

        vals = (

            energy[
                species
                .str
                .lower()
                ==
                sp.lower()
            ]

            .dropna()

            .to_numpy(
                float
            )
        )

        if len(
            vals
        ) == 0:
            continue

        names.append(
            sp
        )

        values.append(
            vals
        )

        colors.append(
            color
        )

    if not values:

        raise RuntimeError(
            "No C/B/Fe insertion energies were found."
        )

    positions = np.arange(
        1,
        len(values)+1,
    )

    bp = ax.boxplot(

        values,

        positions=positions,

        widths=0.50,

        patch_artist=True,

        showfliers=False,

        medianprops=dict(
            color=DARK,
            linewidth=1.5,
        ),

        whiskerprops=dict(
            color=GRAY,
            linewidth=1.0,
        ),

        capprops=dict(
            color=GRAY,
            linewidth=1.0,
        ),

        boxprops=dict(
            linewidth=1.1,
        ),
    )

    for box, color in zip(
        bp[
            "boxes"
        ],
        colors,
    ):

        box.set_facecolor(
            color
        )

        box.set_alpha(
            0.62
        )

        box.set_edgecolor(
            color
        )

    # Actual event values overlaid with deterministic jitter.
    rng = np.random.default_rng(
        7
    )

    for x0, vals, color in zip(
        positions,
        values,
        colors,
    ):

        jitter = rng.uniform(
            -0.11,
            0.11,
            size=len(
                vals
            ),
        )

        ax.scatter(

            x0
            +
            jitter,

            vals,

            s=12,

            c=color,

            alpha=0.36,

            edgecolors="none",

            zorder=3,
        )

        ax.text(

            x0,

            0.98,

            rf"$n={len(vals)}$",

            transform=ax.get_xaxis_transform(),

            ha="center",

            va="top",

            fontsize=9.0,

            fontweight="bold",

            color=GRAY,
        )

    ax.axhline(

        0,

        color=LIGHT_GRAY,

        lw=1.0,

        zorder=0,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        names,
        fontweight="bold",
    )

    ax.set_xlabel(
        "Incorporated species",
        fontweight="bold",
    )

    ax.set_ylabel(
        r"Unrelaxed insertion energy, "
        r"$\Delta E_{\rm ins}$ (eV)",
        fontweight="bold",
    )

    ax.set_title(
        "Species-resolved energetics",
        fontweight="bold",
        pad=9,
    )

    style(
        ax
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

    required = [
        CSV_FILE,
        INITIAL_XYZ,
    ]

    missing = [
        p
        for p
        in required
        if not p.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Missing required files:\n"
            +
            "\n".join(
                str(p)
                for p
                in missing
            )
        )

    df = pd.read_csv(
        CSV_FILE
    )

    if len(
        df
    ) == 0:

        raise RuntimeError(
            f"CSV is empty: {CSV_FILE}"
        )

    if "n_atoms" not in df.columns:

        raise RuntimeError(
            f"{CSV_FILE} does not contain n_atoms.\n"
            f"Columns are:\n{list(df.columns)}"
        )

    # -------------------------------------------------------------------------
    # Initial/final structures
    # -------------------------------------------------------------------------

    sym_initial, xyz_initial = read_xyz(
        INITIAL_XYZ
    )

    n_initial = (

        int(
            df[
                "n_initial"
            ]
            .iloc[0]
        )

        if
        "n_initial"
        in df.columns

        else
        len(
            xyz_initial
        )
    )

    expected_final_n = int(
        df[
            "n_atoms"
        ]
        .iloc[-1]
    )

    final_path, structure_source = choose_final_structure(
        expected_final_n
    )

    symbols, xyz = read_xyz(
        final_path
    )

    if len(
        xyz
    ) < n_initial:

        raise RuntimeError(
            f"Final structure N={len(xyz)} "
            f"is smaller than n_initial={n_initial}."
        )

    # -------------------------------------------------------------------------
    # Real structural bonds
    # -------------------------------------------------------------------------

    bonds = build_bonds(
        symbols,
        xyz,
    )

    layer_centers = two_layer_centers(
        xyz,
        symbols,
        n_initial,
    )

    # =========================================================================
    # FIGURE
    # =========================================================================

    fig, axes = plt.subplots(

        2,
        2,

        figsize=(
            15.6,
            10.5,
        ),

        facecolor="white",
    )

    plt.subplots_adjust(

        left=0.075,
        right=0.975,

        bottom=0.075,
        top=0.95,

        wspace=0.24,
        hspace=0.30,
    )

    # =========================================================================
    # PANEL (a): TOP VIEW
    # =========================================================================

    ax = axes[
        0,
        0
    ]

    panel(
        ax,
        "a",
    )

    draw_structure(

        ax,

        symbols,
        xyz,

        n_initial,

        projection="xy",

        bonds=bonds,
    )

    ax.set_xlabel(
        r"$x$ (Å)",
        fontweight="bold",
    )

    ax.set_ylabel(
        r"$y$ (Å)",
        fontweight="bold",
    )

    ax.set_title(
        "Final structure — top view",
        fontweight="bold",
        pad=9,
    )

    # One compact legend only.
    legend_handles = [

        Line2D(
            [0], [0],
            marker="o",
            ls="",
            ms=7,
            mfc=C_INITIAL,
            mec="none",
            label="initial graphene C",
        ),

        Line2D(
            [0], [0],
            marker="o",
            ls="",
            ms=7,
            mfc=C_ADDED,
            mec=WHITE,
            label="added C",
        ),

        Line2D(
            [0], [0],
            marker="o",
            ls="",
            ms=7,
            mfc=B_COLOR,
            mec=WHITE,
            label="B",
        ),

        Line2D(
            [0], [0],
            marker="o",
            ls="",
            ms=7,
            mfc=FE_COLOR,
            mec=WHITE,
            label="Fe",
        ),
    ]

    ax.legend(

        handles=legend_handles,

        frameon=False,

        loc="best",

        ncol=2,

        fontsize=8.5,

        columnspacing=0.8,

        handletextpad=0.35,
    )

    # =========================================================================
    # PANEL (b): SIDE VIEW
    # =========================================================================

    ax = axes[
        0,
        1
    ]

    panel(
        ax,
        "b",
    )

    draw_structure(

        ax,

        symbols,
        xyz,

        n_initial,

        projection="xz",

        bonds=bonds,
    )

    # Initial bilayer positions as subtle guides.
    for zc in layer_centers:

        ax.axhline(

            zc,

            color=LIGHT_GRAY,

            lw=1.0,

            ls="--",

            alpha=0.70,

            zorder=0,
        )

    ax.set_xlabel(
        r"$x$ (Å)",
        fontweight="bold",
    )

    ax.set_ylabel(
        r"$z$ (Å)",
        fontweight="bold",
    )

    ax.set_title(
        "Finite bilayer graphene — side view",
        fontweight="bold",
        pad=9,
    )

    # =========================================================================
    # PANEL (c)
    # =========================================================================

    plot_event_sequence(

        axes[
            1,
            0
        ],

        df,
    )

    # =========================================================================
    # PANEL (d)
    # =========================================================================

    plot_energy_by_species(

        axes[
            1,
            1
        ],

        df,
    )

    # =========================================================================
    # SUMMARY CSV
    # =========================================================================

    scol = species_column(
        df
    )

    ecol = find_column(
        df,
        exact=(
            "e_ads_unrelaxed_eV",
            "e_ads_unrelaxed",
        ),
        contains=(
            "e_ads_unrelaxed",
            "eads_unrelaxed",
        ),
    )

    if scol is not None:

        event_species = (

            df[
                scol
            ]

            .astype(
                str
            )

            .str
            .strip()
        )

    else:

        event_species = pd.Series(
            [],
            dtype=str,
        )

    rows = []

    for sp in (
        "C",
        "B",
        "Fe",
    ):

        mask = (

            event_species
            .str
            .lower()
            ==
            sp.lower()
        )

        n_events = int(
            mask.sum()
        )

        median_energy = np.nan
        mean_energy = np.nan

        if (
            ecol is not None
            and
            n_events > 0
        ):

            vals = (

                pd.to_numeric(
                    df.loc[
                        mask,
                        ecol
                    ],
                    errors="coerce",
                )

                .dropna()

                .to_numpy(
                    float
                )
            )

            if len(
                vals
            ):

                median_energy = float(
                    np.median(
                        vals
                    )
                )

                mean_energy = float(
                    np.mean(
                        vals
                    )
                )

        rows.append({
            "species":
            sp,

            "n_events":
            n_events,

            "event_fraction":
            (
                n_events
                /
                len(df)
                if len(df)
                else np.nan
            ),

            "mean_unrelaxed_insertion_energy_eV":
            mean_energy,

            "median_unrelaxed_insertion_energy_eV":
            median_energy,
        })

    # Structure-level information appended as extra columns.
    summary = pd.DataFrame(
        rows
    )

    summary[
        "n_initial_atoms"
    ] = (
        n_initial
    )

    summary[
        "n_final_atoms"
    ] = (
        len(
            xyz
        )
    )

    summary[
        "n_growth_events"
    ] = (
        len(
            df
        )
    )

    summary[
        "structure_source"
    ] = (
        structure_source
    )

    if len(
        layer_centers
    ) >= 2:

        summary[
            "initial_bilayer_spacing_A"
        ] = (

            layer_centers[1]
            -
            layer_centers[0]
        )

    else:

        summary[
            "initial_bilayer_spacing_A"
        ] = np.nan

    summary.to_csv(

        HERE
        /
        f"{OUT}_summary.csv",

        index=False,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    out = (
        HERE
        /
        OUT
    )

    fig.savefig(

        out.with_suffix(
            ".pdf"
        ),

        bbox_inches="tight",

        pad_inches=0.08,
    )

    fig.savefig(

        out.with_suffix(
            ".svg"
        ),

        bbox_inches="tight",

        pad_inches=0.08,
    )

    fig.savefig(

        out.with_suffix(
            ".png"
        ),

        dpi=600,

        bbox_inches="tight",

        pad_inches=0.08,
    )

    # =========================================================================
    # REPORT
    # =========================================================================

    print(
        "\nFigure 7 generated:"
    )

    print(
        " ",
        out.with_suffix(
            ".pdf"
        )
    )

    print(
        " ",
        out.with_suffix(
            ".svg"
        )
    )

    print(
        " ",
        out.with_suffix(
            ".png"
        )
    )

    print(

        " ",

        HERE
        /
        f"{OUT}_summary.csv"
    )

    print(
        "\nInput:"
    )

    print(
        f"  CSV: {CSV_FILE}"
    )

    print(
        f"  Initial XYZ: {INITIAL_XYZ}"
    )

    print(
        f"  Final structure: {final_path}"
    )

    print(
        f"  Structure source: {structure_source}"
    )

    print(
        f"  N initial: {n_initial}"
    )

    print(
        f"  N final: {len(xyz)}"
    )

    print(
        f"  Accepted events: {len(df)}"
    )

    if len(
        layer_centers
    ) >= 2:

        print(

            "  Initial graphene layer centers: "
            f"{layer_centers[0]:.4f}, "
            f"{layer_centers[1]:.4f} Å"

        )

        print(

            "  Initial bilayer separation: "
            f"{layer_centers[1]-layer_centers[0]:.4f} Å"

        )

    if scol is not None:

        print(
            "\nAccepted-event counts:"
        )

        for sp in (
            "C",
            "B",
            "Fe",
        ):

            nsp = int(

                (
                    event_species
                    .str
                    .lower()
                    ==
                    sp.lower()
                )
                .sum()

            )

            print(
                f"  {sp}: {nsp}"
            )

    plt.show()


if __name__ == "__main__":
    main()
