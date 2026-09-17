#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Results Fig. 1 — GPAW growth of Si(100), Si(110), Si(111)

Place this file at:
    PAPER/figure1/figure1_silicon.py

It reads:
    PAPER/silicon/100/
    PAPER/silicon/110/
    PAPER/silicon/111/

Panel (a) is a 3D ball-and-stick representation built directly from the
actual XYZ coordinates. Initial Si atoms are shown in silver/gray and atoms
incorporated during growth are highlighted by facet-specific colors.

If seed_01/, seed_02/, ... subdirectories containing grafeno_cutoff.csv
are later added under each orientation, panels (b)--(d) automatically use
ensemble mean ± standard deviation.

Outputs
-------
Figure2_Si_GPAW_results.pdf
Figure2_Si_GPAW_results.svg
Figure2_Si_GPAW_results.png
Figure2_Si_GPAW_results_summary.csv

Dependencies
------------
numpy
pandas
matplotlib

ASE is NOT required for this plotting script.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import gridspec
from matplotlib.lines import Line2D


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "results"
SIROOT = PAPER / "silicon"
OUT = "Figure2_Si_GPAW_results"


# =============================================================================
# SYSTEMS AND COLORS
# =============================================================================

ORIS = ("100", "110", "111")

LABEL = {
    "100": "Si(100)",
    "110": "Si(110)",
    "111": "Si(111)",
}

COLOR = {
    "100": "#1F77B4",
    "110": "#D97706",
    "111": "#2A9D8F",
}

GRAY = "#6B7280"
DARK = "#18212B"
INITIAL = "#AEB5BD"
INITIAL_EDGE = "#737B84"
BOND_INITIAL = "#A7ADB4"
BOND_GROWN = "#4B5563"
LIGHT = "#D1D5DB"
WHITE = "#FFFFFF"


# =============================================================================
# VISUAL PARAMETERS FOR PANEL (a)
# =============================================================================

# Si first-neighbor visualization cutoff.
SI_BOND_CUTOFF_A = 2.65

# Only the upper part of the original slab is displayed.
# All atoms incorporated during growth are always displayed.
INITIAL_DISPLAY_DEPTH_A = 7.5

# Common 3D camera for all orientations.
VIEW_ELEV = 19
VIEW_AZIM = -58


# =============================================================================
# FIGURE STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 1.4,
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 5,
    "ytick.major.size": 5,

    # Keep text editable in Illustrator/Inkscape.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# =============================================================================
# PLOT HELPERS
# =============================================================================

def panel(ax, letter, x=-0.13, y=1.10):
    """Large bold panel label."""
    if hasattr(ax, "text2D"):
        ax.text2D(
            x, y, letter,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
            clip_on=False,
        )
    else:
        ax.text(
            x, y, letter,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            ha="left",
            va="top",
            color="black",
            clip_on=False,
        )


def style(ax):
    """Clean npj-style 2D axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


# =============================================================================
# XYZ READER
# =============================================================================

def read_xyz(path):
    """
    Read a standard/extended XYZ without ASE.

    Returns
    -------
    symbols : ndarray
    xyz     : ndarray (N,3)
    """

    path = Path(path)

    lines = path.read_text(
        errors="replace"
    ).splitlines()

    if not lines:
        raise ValueError(
            f"Empty XYZ file: {path}"
        )

    n = int(
        lines[0].strip()
    )

    sym = []
    xyz = []

    for line in lines[2:2+n]:

        f = line.split()

        if len(f) < 4:
            continue

        sym.append(
            f[0]
        )

        xyz.append([
            float(f[1]),
            float(f[2]),
            float(f[3]),
        ])

    if len(xyz) != n:

        raise ValueError(
            f"XYZ mismatch in {path}: "
            f"expected {n}, read {len(xyz)}"
        )

    return (
        np.asarray(sym, object),
        np.asarray(xyz, float),
    )


def xyz_n(path):
    """Return number of atoms written in an XYZ header."""

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


# =============================================================================
# READ CRYSTALLOGRAPHIC INFORMATION FROM LOG
# =============================================================================

def parse_geometry(log):
    """
    Extract:

        p = structural motif period
        q = gap period
        gaps
        N_template
    """

    m = re.search(

        r"Template extend-columns:.*?"
        r"periodo_colunas=(\d+)\([^)]*\).*?"
        r"periodo_gaps=(\d+)\([^)]*\).*?"
        r"gaps=\[([^\]]+)\].*?"
        r"N_template=(\d+)",

        log,
    )

    if not m:

        return (
            None,
            None,
            [],
            None,
        )

    return (

        int(
            m.group(1)
        ),

        int(
            m.group(2)
        ),

        [
            float(x.strip())
            for x
            in m.group(3).split(",")
        ],

        int(
            m.group(4)
        ),
    )


# =============================================================================
# PROVENANCE CONTROL
# =============================================================================

def consistent_structure(
    run_dir,
    expected_n,
):
    """
    Choose a structure consistent with the final atom count in the CSV.

    Priority
    --------
    1. grafeno_cutoff_final.xyz
    2. grafeno_cutoff_checkpoint.xyz

    This automatically handles the current Si(110) directory where the
    checkpoint matches the CSV but the archived final.xyz belongs to another
    state/run.
    """

    candidates = [

        (
            run_dir /
            "grafeno_cutoff_final.xyz",

            "final.xyz",
        ),

        (
            run_dir /
            "grafeno_cutoff_checkpoint.xyz",

            "checkpoint.xyz",
        ),
    ]

    for p, name in candidates:

        if (
            p.exists()
            and
            xyz_n(p) == expected_n
        ):

            return (
                p,
                name,
            )

    existing = [

        (
            p,
            name,
            xyz_n(p),
        )

        for p, name
        in candidates

        if p.exists()
    ]

    existing = [
        x
        for x in existing
        if x[2] is not None
    ]

    if not existing:

        raise FileNotFoundError(
            f"No final/checkpoint XYZ in {run_dir}"
        )

    p, name, n = min(

        existing,

        key=lambda x:
        abs(
            x[2] - expected_n
        ),
    )

    warnings.warn(

        f"{run_dir}: no XYZ matches CSV N={expected_n}; "
        f"using {name} with N={n}. "
        f"Fix provenance before publication."

    )

    return (
        p,
        name + " [COUNT MISMATCH]",
    )


# =============================================================================
# LOAD REPRESENTATIVE RUN
# =============================================================================

def load_base(ori):

    d = SIROOT / ori

    csvfile = (
        d /
        "grafeno_cutoff.csv"
    )

    logfile = (
        d /
        "grafeno_cutoff.log"
    )

    xyz_initial = (
        d /
        "grafeno_cutoff_initial.xyz"
    )

    for required in (
        csvfile,
        logfile,
        xyz_initial,
    ):

        if not required.exists():

            raise FileNotFoundError(
                f"Missing required file: {required}"
            )

    df = pd.read_csv(
        csvfile
    )

    log = logfile.read_text(
        errors="replace"
    )

    s0, x0 = read_xyz(
        xyz_initial
    )

    expected = int(
        df["n_atoms"].iloc[-1]
    )

    final_path, final_source = (
        consistent_structure(
            d,
            expected,
        )
    )

    sf, xf = read_xyz(
        final_path
    )

    p, q, gaps, nt = (
        parse_geometry(
            log
        )
    )

    return {

        "dir": d,

        "df": df,
        "log": log,

        "sym0": s0,
        "xyz0": x0,

        "symf": sf,
        "xyzf": xf,

        "final_path": final_path,
        "final_source": final_source,

        "p": p,
        "q": q,
        "gaps": gaps,

        "ntemplate": nt,
    }


# =============================================================================
# MULTIPLE RANDOM SEEDS
# =============================================================================

def seed_csvs(ori):
    """
    Automatically detect future independent runs.

    Example:

    PAPER/silicon/100/
        seed_01/grafeno_cutoff.csv
        seed_02/grafeno_cutoff.csv
        seed_03/grafeno_cutoff.csv

    If >= 2 such files exist, panels b-d use the ensemble.
    """

    d = SIROOT / ori

    found = []

    for pat in (

        "seed*/grafeno_cutoff.csv",
        "seed_*/grafeno_cutoff.csv",

        "run*/grafeno_cutoff.csv",
        "rep*/grafeno_cutoff.csv",

    ):

        found += list(
            d.glob(pat)
        )

    found = sorted(

        set(
            p.resolve()
            for p in found
        )
    )

    if len(found) >= 2:

        return found

    return [

        (
            d /
            "grafeno_cutoff.csv"
        ).resolve()

    ]


def seed_dfs(ori):

    return [

        pd.read_csv(p)

        for p
        in seed_csvs(ori)

    ]


# =============================================================================
# ENSEMBLE STATISTICS
# =============================================================================

def ensemble_curve(
    dfs,
    metric,
    ngrid=220,
):
    """
    Interpolate accepted trajectories onto normalized growth progress.

    Returns
    -------
    x
    mean
    std
    """

    xg = np.linspace(
        0,
        1,
        ngrid,
    )

    curves = []

    for df in dfs:

        if (
            metric not in df
            or
            len(df) == 0
        ):

            continue

        y = np.asarray(
            df[metric],
            float,
        )

        x = np.linspace(
            0,
            1,
            len(y),
        )

        ok = np.isfinite(
            y
        )

        if ok.sum() == 0:
            continue

        if ok.sum() == 1:

            yi = np.full_like(
                xg,
                y[ok][0],
            )

        else:

            yi = np.interp(
                xg,
                x[ok],
                y[ok],
            )

        curves.append(
            yi
        )

    if not curves:

        raise RuntimeError(
            f"No valid data for metric {metric}"
        )

    arr = np.vstack(
        curves
    )

    mean = arr.mean(
        axis=0
    )

    if len(arr) > 1:

        sd = arr.std(
            axis=0,
            ddof=1,
        )

    else:

        sd = np.zeros_like(
            mean
        )

    return (
        xg,
        mean,
        sd,
    )


def final_stats(
    dfs,
    metric,
):

    values = np.asarray([

        float(
            df[metric].iloc[-1]
        )

        for df in dfs

        if metric in df

    ], float)

    if len(values) == 0:

        raise RuntimeError(
            f"No final values for metric {metric}"
        )

    mean = values.mean()

    if len(values) > 1:

        sd = values.std(
            ddof=1
        )

    else:

        sd = 0.0

    return (
        mean,
        sd,
    )


def gap_text(gaps):

    if len(gaps) == 0:

        return "n/a"

    if len(gaps) == 1:

        return (
            f"{gaps[0]:.3f} Å"
        )

    return (

        " / ".join(

            f"{g:.3f}"

            for g
            in gaps

        )

        + " Å"
    )


# =============================================================================
# 3D CRYSTAL VISUALIZATION
# =============================================================================

def bond_pairs(
    xyz,
    cutoff=SI_BOND_CUTOFF_A,
):
    """
    Determine Si-Si bonds for visualization.

    Periodic replicas are deliberately not reconstructed because they clutter
    the crystal rendering. The purpose here is to show the morphology and
    distinction between the original crystal and newly incorporated atoms.
    """

    xyz = np.asarray(
        xyz,
        float,
    )

    n = len(
        xyz
    )

    pairs = []

    for i in range(
        n - 1
    ):

        delta = (
            xyz[i+1:]
            -
            xyz[i]
        )

        d2 = np.einsum(
            "ij,ij->i",
            delta,
            delta,
        )

        js = np.where(

            (d2 > 1e-8)
            &
            (d2 <= cutoff**2)

        )[0]

        for j0 in js:

            pairs.append(

                (
                    i,
                    i + 1 + int(j0),
                )

            )

    return pairs


def equalize_3d_box(
    ax,
    xyz,
    margin=0.55,
):
    """
    Give x/y/z approximately identical metric scaling.
    """

    mins = np.min(
        xyz,
        axis=0,
    )

    maxs = np.max(
        xyz,
        axis=0,
    )

    centers = (
        0.5
        *
        (
            mins
            +
            maxs
        )
    )

    spans = (
        maxs
        -
        mins
    )

    radius = (

        max(
            float(
                np.max(spans)
            )
            *
            0.52,

            1.0,
        )

        +
        margin
    )

    ax.set_xlim(
        centers[0] - radius,
        centers[0] + radius,
    )

    ax.set_ylim(
        centers[1] - radius,
        centers[1] + radius,
    )

    ax.set_zlim(
        centers[2] - radius,
        centers[2] + radius,
    )

    try:

        ax.set_box_aspect(
            (1, 1, 1)
        )

    except Exception:

        pass


def plot_crystal_3d(
    ax,
    ori,
    run,
    initial_depth=INITIAL_DISPLAY_DEPTH_A,
):
    """
    Draw the actual Si crystal.

    Silver
    ------
    Atoms already present in the initial slab.

    Facet color
    -----------
    Atoms incorporated during the growth simulation.
    """

    n0 = len(
        run["xyz0"]
    )

    xyzf = np.asarray(
        run["xyzf"],
        float,
    )

    if len(xyzf) < n0:

        raise ValueError(

            f"{LABEL[ori]}: "
            "final structure has fewer atoms "
            "than the initial structure."

        )

    # Initial crystal growth front.
    zfront0 = float(

        np.max(
            run["xyz0"][:, 2]
        )

    )

    initial_mask = (
        np.arange(
            len(xyzf)
        )
        <
        n0
    )

    added_mask = (
        ~initial_mask
    )

    # Display near-surface initial atoms and every grown atom.
    display_mask = (

        added_mask

        |

        (
            initial_mask
            &
            (
                xyzf[:, 2]
                >=
                zfront0 - initial_depth
            )
        )

    )

    display_indices = np.where(
        display_mask
    )[0]

    shown = (
        xyzf[
            display_indices
        ]
        .copy()
    )

    # Set original growth front at z=0.
    shown[:, 2] -= (
        zfront0
    )

    local_of_global = {

        int(global_i):
        int(local_i)

        for local_i, global_i
        in enumerate(
            display_indices
        )
    }

    # -------------------------------------------------------------------------
    # Si-Si bonds
    # -------------------------------------------------------------------------

    for i, j in bond_pairs(

        xyzf,

        cutoff=SI_BOND_CUTOFF_A,

    ):

        if (

            i not in local_of_global

            or

            j not in local_of_global

        ):

            continue

        li = (
            local_of_global[i]
        )

        lj = (
            local_of_global[j]
        )

        p1 = shown[li]
        p2 = shown[lj]

        grown_bond = (

            i >= n0
            or
            j >= n0

        )

        ax.plot(

            [
                p1[0],
                p2[0],
            ],

            [
                p1[1],
                p2[1],
            ],

            [
                p1[2],
                p2[2],
            ],

            color=(
                BOND_GROWN
                if grown_bond
                else BOND_INITIAL
            ),

            lw=(
                1.05
                if grown_bond
                else 0.72
            ),

            alpha=(
                0.68
                if grown_bond
                else 0.42
            ),

            zorder=1,
        )

    local_initial = np.array([

        display_indices[k]
        <
        n0

        for k
        in range(
            len(display_indices)
        )

    ], dtype=bool)

    local_added = (
        ~local_initial
    )

    # -------------------------------------------------------------------------
    # Initial atoms
    # -------------------------------------------------------------------------

    if np.any(
        local_initial
    ):

        ax.scatter(

            shown[
                local_initial,
                0
            ],

            shown[
                local_initial,
                1
            ],

            shown[
                local_initial,
                2
            ],

            s=54,

            c=INITIAL,

            edgecolors=INITIAL_EDGE,

            linewidths=0.55,

            alpha=0.88,

            depthshade=True,

            zorder=3,
        )

    # -------------------------------------------------------------------------
    # Grown atoms
    # -------------------------------------------------------------------------

    if np.any(
        local_added
    ):

        ax.scatter(

            shown[
                local_added,
                0
            ],

            shown[
                local_added,
                1
            ],

            shown[
                local_added,
                2
            ],

            s=76,

            c=COLOR[ori],

            edgecolors=WHITE,

            linewidths=0.85,

            alpha=1.0,

            depthshade=True,

            zorder=5,
        )

    # -------------------------------------------------------------------------
    # Initial growth-front frame
    # -------------------------------------------------------------------------

    xmin, ymin = (
        shown[:, :2]
        .min(
            axis=0
        )
    )

    xmax, ymax = (
        shown[:, :2]
        .max(
            axis=0
        )
    )

    pad_x = (
        0.02
        *
        max(
            xmax - xmin,
            1.0,
        )
    )

    pad_y = (
        0.02
        *
        max(
            ymax - ymin,
            1.0,
        )
    )

    xx = [

        xmin - pad_x,
        xmax + pad_x,
        xmax + pad_x,
        xmin - pad_x,
        xmin - pad_x,

    ]

    yy = [

        ymin - pad_y,
        ymin - pad_y,
        ymax + pad_y,
        ymax + pad_y,
        ymin - pad_y,

    ]

    zz = np.zeros(
        5
    )

    ax.plot(

        xx,
        yy,
        zz,

        color=DARK,

        lw=1.0,

        ls="--",

        alpha=0.55,

        zorder=2,
    )

    # -------------------------------------------------------------------------
    # +z arrow
    # -------------------------------------------------------------------------

    xr = max(
        xmax - xmin,
        1.0,
    )

    yr = max(
        ymax - ymin,
        1.0,
    )

    zmax = float(
        np.max(
            shown[:, 2]
        )
    )

    arrow_x = (
        xmin
        -
        0.06 * xr
    )

    arrow_y = (
        ymin
        -
        0.06 * yr
    )

    arrow_z0 = max(

        float(
            np.min(
                shown[:, 2]
            )
        ),

        -0.55
        *
        initial_depth,

    )

    arrow_dz = max(

        2.0,

        0.34
        *
        (
            zmax
            -
            arrow_z0
            +
            1.0
        ),

    )

    try:

        ax.quiver(

            arrow_x,
            arrow_y,
            arrow_z0,

            0,
            0,
            arrow_dz,

            color=DARK,

            linewidth=1.7,

            arrow_length_ratio=0.16,
        )

        ax.text(

            arrow_x,
            arrow_y,
            arrow_z0
            +
            arrow_dz * 1.08,

            "+z",

            fontsize=9.5,

            fontweight="bold",

            color=DARK,

            ha="center",
        )

    except Exception:

        pass

    # -------------------------------------------------------------------------
    # Method-linked crystallographic descriptors
    # -------------------------------------------------------------------------

    ax.text2D(

        0.02,
        0.975,

        rf"$\mathbf{{p={run['p']}}}$"
        +
        "\n"
        +
        rf"$\Delta z$: {gap_text(run['gaps'])}",

        transform=ax.transAxes,

        ha="left",
        va="top",

        fontsize=9.2,

        fontweight="bold",

        color=DARK,

        bbox=dict(

            boxstyle="round,pad=0.24",

            fc="white",

            ec=LIGHT,

            alpha=0.90,
        ),

        zorder=20,
    )

    # -------------------------------------------------------------------------
    # Atom counts
    # -------------------------------------------------------------------------

    ax.text2D(

        0.98,
        0.045,

        rf"$N_0={n0}$"
        +
        "\n"
        +
        rf"$N_{{\rm add}}="
        rf"{int(run['df']['n_added'].iloc[-1])}$",

        transform=ax.transAxes,

        ha="right",
        va="bottom",

        fontsize=9.2,

        fontweight="bold",

        color=DARK,

        bbox=dict(

            boxstyle="round,pad=0.16",

            fc="white",

            ec="none",

            alpha=0.72,
        ),

        zorder=20,
    )

    # -------------------------------------------------------------------------
    # Facet title
    # -------------------------------------------------------------------------

    ax.text2D(

        0.50,
        1.015,

        LABEL[ori],

        transform=ax.transAxes,

        ha="center",
        va="bottom",

        fontsize=13.5,

        fontweight="bold",

        color=DARK,

        clip_on=False,
    )

    # Same camera for all three orientations.
    ax.view_init(

        elev=VIEW_ELEV,

        azim=VIEW_AZIM,
    )

    try:

        ax.set_proj_type(
            "ortho"
        )

    except Exception:

        pass

    equalize_3d_box(
        ax,
        shown,
    )

    ax.set_axis_off()


# =============================================================================
# MAIN
# =============================================================================

def main():

    if not SIROOT.exists():

        raise FileNotFoundError(

            f"Cannot find {SIROOT}.\n"
            "Put this script inside PAPER/figure1/."

        )

    # Representative structures
    base = {

        o:
        load_base(o)

        for o
        in ORIS
    }

    # Statistical ensembles
    ens = {

        o:
        seed_dfs(o)

        for o
        in ORIS
    }

    # =========================================================================
    # FIGURE CANVAS
    # =========================================================================

    fig = plt.figure(

        figsize=(
            16.5,
            11.8,
        ),

        facecolor="white",
    )

    outer = gridspec.GridSpec(

        2,
        2,

        figure=fig,

        left=0.065,
        right=0.985,

        bottom=0.095,
        top=0.945,

        wspace=0.22,
        hspace=0.30,
    )

    # =========================================================================
    # PANEL (a)
    # ACTUAL Si CRYSTALS
    # =========================================================================

    sub = gridspec.GridSpecFromSubplotSpec(

        1,
        3,

        subplot_spec=outer[0, 0],

        wspace=-0.03,
    )

    axs3d = []

    for i, o in enumerate(
        ORIS
    ):

        ax3 = fig.add_subplot(

            sub[0, i],

            projection="3d",
        )

        axs3d.append(
            ax3
        )

        plot_crystal_3d(

            ax3,

            o,

            base[o],
        )

    panel(

        axs3d[0],

        "a",

        x=-0.16,

        y=1.11,
    )

    axs3d[1].text2D(

        0.50,
        1.115,

        "Facet-resolved first-principles Si growth",

        transform=axs3d[1].transAxes,

        ha="center",
        va="bottom",

        fontsize=15.5,

        fontweight="bold",

        color=DARK,

        clip_on=False,
    )

    legend_handles = [

        Line2D(

            [0],
            [0],

            marker="o",

            ls="",

            ms=8,

            mfc=INITIAL,

            mec=INITIAL_EDGE,

            label="initial Si",
        ),

        Line2D(

            [0],
            [0],

            marker="o",

            ls="",

            ms=8,

            mfc="#3B82A0",

            mec="white",

            label="incorporated Si (facet color)",
        ),
    ]

    axs3d[1].legend(

        handles=legend_handles,

        loc="lower center",

        bbox_to_anchor=(
            0.50,
            -0.075,
        ),

        frameon=False,

        fontsize=8.2,

        ncol=2,

        handletextpad=0.38,

        columnspacing=0.9,

        borderaxespad=0.0,
    )

    # =========================================================================
    # PANEL (b)
    # DFT INSERTION ENERGY
    # =========================================================================

    ax = fig.add_subplot(
        outer[0, 1]
    )

    panel(
        ax,
        "b",
    )

    for o in ORIS:

        x, m, s = ensemble_curve(

            ens[o],

            "e_ads_unrelaxed_eV",
        )

        ax.plot(

            x,
            m,

            color=COLOR[o],

            lw=2.6,

            label=LABEL[o],
        )

        if len(
            ens[o]
        ) > 1:

            ax.fill_between(

                x,

                m - s,

                m + s,

                color=COLOR[o],

                alpha=0.17,

                lw=0,
            )

        emean = (

            base[o]["df"]
            ["e_ads_unrelaxed_eV"]
            .mean()

        )

        ax.axhline(

            emean,

            color=COLOR[o],

            lw=1.0,

            ls=":",

            alpha=0.65,
        )

    ax.axhline(

        0,

        color=LIGHT,

        lw=1.3,
    )

    ax.set_xlabel(

        "Normalized accepted-growth progress",

        fontweight="bold",
    )

    ax.set_ylabel(

        r"Unrelaxed insertion energy, "
        r"$\Delta E_{\rm ins}$ (eV)",

        fontweight="bold",
    )

    ax.set_title(

        "Energetic selection along the DFT growth trajectory",

        fontweight="bold",

        pad=10,
    )

    ax.legend(

        frameon=False,

        loc="best",
    )

    if max(

        len(
            ens[o]
        )

        for o
        in ORIS

    ) > 1:

        ax.text(

            0.02,
            0.04,

            "Lines: ensemble mean; shading: ±1 s.d.",

            transform=ax.transAxes,

            fontsize=9.5,

            color=GRAY,
        )

    style(
        ax
    )

    # =========================================================================
    # PANEL (c)
    # FOURFOLD COORDINATION
    # =========================================================================

    ax = fig.add_subplot(
        outer[1, 0]
    )

    panel(
        ax,
        "c",
    )

    for o in ORIS:

        x, m, s = ensemble_curve(

            ens[o],

            "fraction_added_CN4",
        )

        ax.plot(

            x,
            m,

            color=COLOR[o],

            lw=2.6,

            label=LABEL[o],
        )

        if len(
            ens[o]
        ) > 1:

            ax.fill_between(

                x,

                np.clip(
                    m - s,
                    0,
                    1,
                ),

                np.clip(
                    m + s,
                    0,
                    1,
                ),

                color=COLOR[o],

                alpha=0.17,

                lw=0,
            )

        yfinal = (

            base[o]["df"]
            ["fraction_added_CN4"]
            .iloc[-1]

        )

        ax.scatter(

            [1],

            [yfinal],

            s=75,

            c=COLOR[o],

            ec="white",

            lw=0.8,

            zorder=5,
        )

    ax.axhline(

        1,

        color=GRAY,

        lw=1.3,

        ls="--",

        alpha=0.75,
    )

    ax.text(

        0.98,
        0.965,

        "ideal tetrahedral fraction",

        transform=ax.transAxes,

        fontsize=9.3,

        color=GRAY,

        ha="right",

        va="top",
    )

    ax.set_ylim(
        -0.03,
        1.05,
    )

    ax.set_xlabel(

        "Normalized accepted-growth progress",

        fontweight="bold",
    )

    ax.set_ylabel(

        r"Fraction of added Si with $CN=4$",

        fontweight="bold",
    )

    ax.set_title(

        "Emergence of bulk-like tetrahedral coordination",

        fontweight="bold",

        pad=10,
    )

    style(
        ax
    )

    # =========================================================================
    # PANEL (d)
    # FINAL GROWTH METRICS
    # =========================================================================

    ax = fig.add_subplot(
        outer[1, 1]
    )

    panel(
        ax,
        "d",
    )

    metrics = [

        (
            "template_growth_occupancy_fraction",

            "Template-site occupancy",

            "#CADBED",

            "#477BA8",
        ),

        (
            "fraction_added_CN4",

            r"Added-Si $CN=4$ fraction",

            "#D9EAD3",

            "#4D8B57",
        ),

        (
            "growth_front_mean_fraction_of_template",

            "Normalized growth-front advance",

            "#F4D6B0",

            "#C87519",
        ),
    ]

    x0 = np.arange(
        3,
        dtype=float,
    )

    w = 0.22

    offsets = (
        -w,
        0,
        w,
    )

    summary = []

    for j, (
        metric,
        lab,
        fc,
        ec,

    ) in enumerate(
        metrics
    ):

        means = []
        sds = []

        for o in ORIS:

            m, s = final_stats(

                ens[o],

                metric,
            )

            means.append(
                m
            )

            sds.append(
                s
            )

        means = np.asarray(
            means
        )

        sds = np.asarray(
            sds
        )

        ax.bar(

            x0 + offsets[j],

            means,

            width=w * 0.90,

            yerr=(
                sds
                if np.any(sds > 0)
                else None
            ),

            capsize=3,

            color=fc,

            edgecolor=ec,

            linewidth=1.3,

            label=lab,

            zorder=3,
        )

        for xb, yb in zip(

            x0 + offsets[j],

            means,

        ):

            ax.text(

                xb,

                yb + 0.025,

                f"{yb:.2f}",

                ha="center",

                va="bottom",

                fontsize=9,

                fontweight="bold",

                color=DARK,
            )

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------

    for o in ORIS:

        row = {

            "orientation": o,

            "n_seed_runs":
            len(
                ens[o]
            ),
        }

        for (
            metric,
            _,
            _,
            _,

        ) in metrics:

            (
                row[
                    metric
                    +
                    "_mean"
                ],

                row[
                    metric
                    +
                    "_std"
                ],

            ) = final_stats(

                ens[o],

                metric,
            )

        (
            row[
                "roughness_A_mean"
            ],

            row[
                "roughness_A_std"
            ],

        ) = final_stats(

            ens[o],

            "roughness_A",
        )

        row[
            "mean_tested_candidates"
        ] = (

            base[o]["df"]
            ["n_tested_candidates"]
            .mean()

        )

        row[
            "mean_chosen_probability"
        ] = (

            base[o]["df"]
            ["chosen_boltzmann_probability"]
            .mean()

        )

        row[
            "n_added"
        ] = int(

            base[o]["df"]
            ["n_added"]
            .iloc[-1]

        )

        row[
            "motif_period"
        ] = (
            base[o]["p"]
        )

        row[
            "gap_period"
        ] = (
            base[o]["q"]
        )

        row[
            "gaps_A"
        ] = ";".join(

            f"{g:.6f}"

            for g
            in base[o]["gaps"]

        )

        row[
            "structure_source"
        ] = (
            base[o]["final_source"]
        )

        summary.append(
            row
        )

    ax.axhline(

        1,

        color=GRAY,

        lw=1.2,

        ls="--",

        alpha=0.7,
    )

    ax.set_ylim(
        0,
        1.13,
    )

    ax.set_xticks(
        x0
    )

    ax.set_xticklabels(

        [
            LABEL[o]
            for o in ORIS
        ],

        fontweight="bold",
    )

    ax.set_ylabel(

        "Dimensionless growth metric",

        fontweight="bold",
    )

    ax.set_title(

        "Facet-dependent structural completion",

        fontweight="bold",

        pad=10,
    )

    ax.legend(

        frameon=False,

        loc="upper center",

        bbox_to_anchor=(
            0.5,
            1.01,
        ),

        ncol=3,

        fontsize=9.4,
    )

    # Roughness stays as text because Å should not be mixed with
    # dimensionless bars on the same ordinate.
    rough = []

    for o in ORIS:

        m, s = final_stats(

            ens[o],

            "roughness_A",
        )

        if s == 0:

            rough.append(

                rf"{LABEL[o]}: "
                rf"$\sigma_h={m:.2f}$ Å"

            )

        else:

            rough.append(

                rf"{LABEL[o]}: "
                rf"$\sigma_h={m:.2f}\pm{s:.2f}$ Å"

            )

    ax.text(

        0.5,
        0.035,

        "   |   ".join(
            rough
        ),

        transform=ax.transAxes,

        ha="center",

        va="bottom",

        fontsize=8.8,

        fontweight="bold",

        color=GRAY,
    )

    style(
        ax
    )

    # =========================================================================
    # SAVE FIGURE
    # =========================================================================

    out = (
        HERE /
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

    pd.DataFrame(
        summary
    ).to_csv(

        HERE /
        f"{OUT}_summary.csv",

        index=False,
    )

    # =========================================================================
    # TERMINAL REPORT
    # =========================================================================

    print(
        "\nGenerated:"
    )

    for ext in (
        "pdf",
        "svg",
        "png",
    ):

        print(

            " ",

            out.with_suffix(
                "." + ext
            ),
        )

    print(

        " ",

        HERE /
        f"{OUT}_summary.csv",
    )

    print(
        "\nStructures used:"
    )

    for o in ORIS:

        print(

            f"  {LABEL[o]}: "

            f"{base[o]['final_path'].name} "

            f"[{base[o]['final_source']}], "

            f"N={len(base[o]['xyzf'])}, "

            f"p={base[o]['p']}, "

            f"gaps={base[o]['gaps']}"

        )

    print(
        "\nSeed ensembles detected:"
    )

    for o in ORIS:

        print(

            f"  {LABEL[o]}: "
            f"n={len(ens[o])}"

        )

    if all(

        len(
            ens[o]
        )
        ==
        1

        for o
        in ORIS

    ):

        print(

            "\nNOTE: current archive contains one trajectory per facet. "
            "Run independent seeds for publication-level "
            "uncertainty estimates."

        )

    plt.show()


if __name__ == "__main__":

    main()
