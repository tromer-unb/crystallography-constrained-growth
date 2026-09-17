#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 6 — Backend portability for crystallography-constrained Si growth

Place this file at:
    PAPER/figure6/figure6.py

Expected directory tree
-----------------------
PAPER/
└── silicon/
    ├── 100/
    │   ├── grafeno_cutoff.csv        # GPAW reference run
    │   ├── grafeno_cutoff_final.xyz
    │   ├── grafeno_cutoff_checkpoint.xyz
    │   ├── grafeno_cutoff_initial.xyz
    │   ├── CHGNET/
    │   │   ├── grafeno_cutoff.csv
    │   │   ├── grafeno_cutoff_final.xyz
    │   │   └── ...
    │   └── MACE/
    │       ├── grafeno_cutoff.csv
    │       ├── grafeno_cutoff_final.xyz
    │       └── ...
    ├── 110/
    │   └── ...
    └── 111/
        └── ...

Scientific logic
----------------
The three backends are compared using complete growth trajectories started
from the same crystallographic systems and the same random seed. Because the
trajectories may bifurcate after different site choices, the figure does NOT
claim a same-candidate energy benchmark. Instead, it compares:

(a) actual final Si(111) structures produced by GPAW, CHGNet, and MACE;
(b) final structure-based growth metrics across all facets/backends;
(c) evolution of tetrahedral coordination during growth;
(d) size of the candidate space and sharpness of stochastic site selection.

This is appropriate for demonstrating backend portability of the growth
framework. A stricter same-candidate GPAW/CHGNet/MACE energetic-ranking
benchmark can be added later as a separate validation.

Outputs
-------
Figure6_Backend_Portability.pdf
Figure6_Backend_Portability.svg
Figure6_Backend_Portability.png
Figure6_Backend_Portability_summary.csv

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
from matplotlib import gridspec
from matplotlib.lines import Line2D


# =============================================================================
# PATHS
# =============================================================================

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent / "results"
SIROOT = PAPER / "silicon"
OUT = "Figure6_Backend_Portability"


# =============================================================================
# SYSTEMS / BACKENDS
# =============================================================================

FACETS = ("100", "110", "111")
BACKENDS = ("GPAW", "CHGNET", "MACE")

FACET_LABEL = {
    "100": "Si(100)",
    "110": "Si(110)",
    "111": "Si(111)",
}

BACKEND_LABEL = {
    "GPAW": "GPAW",
    "CHGNET": "CHGNet",
    "MACE": "MACE",
}

BACKEND_COLOR = {
    "GPAW": "#2F6B9A",
    "CHGNET": "#2A9D8F",
    "MACE": "#D97706",
}

FACET_MARKER = {
    "100": "o",
    "110": "s",
    "111": "^",
}

DARK = "#17212B"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
INITIAL = "#B7BDC4"
INITIAL_EDGE = "#7B838B"
BOND_INITIAL = "#A8AEB5"
BOND_GROWN = "#59616A"
WHITE = "#FFFFFF"

SI_BOND_CUTOFF_A = 2.68
DISPLAY_INITIAL_DEPTH_A = 8.0


# =============================================================================
# STYLE
# =============================================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.3,
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


def panel(ax, letter, x=-0.12, y=1.08):
    if hasattr(ax, "text2D"):
        ax.text2D(
            x, y, letter,
            transform=ax.transAxes,
            fontsize=23,
            fontweight="bold",
            ha="left", va="top",
            color="black",
            clip_on=False,
        )
    else:
        ax.text(
            x, y, letter,
            transform=ax.transAxes,
            fontsize=23,
            fontweight="bold",
            ha="left", va="top",
            color="black",
            clip_on=False,
        )


# =============================================================================
# FILE HELPERS
# =============================================================================


def backend_dir(facet, backend):
    root = SIROOT / facet
    if backend == "GPAW":
        return root
    return root / backend


def first_existing(directory, names, required=True):
    directory = Path(directory)
    for name in names:
        p = directory / name
        if p.exists():
            return p
    if required:
        raise FileNotFoundError(
            f"None of these files exist in {directory}: {names}"
        )
    return None


def read_xyz(path):
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ: {path}")
    n = int(lines[0].strip())
    symbols = []
    xyz = []
    for line in lines[2:2+n]:
        f = line.split()
        if len(f) < 4:
            continue
        symbols.append(f[0])
        xyz.append([float(f[1]), float(f[2]), float(f[3])])
    if len(xyz) != n:
        raise ValueError(
            f"XYZ mismatch in {path}: expected {n}, read {len(xyz)}"
        )
    return np.asarray(symbols, object), np.asarray(xyz, float)


def xyz_n(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return int(path.read_text(errors="replace").splitlines()[0].strip())
    except Exception:
        return None


def consistent_structure(run_dir, expected_n):
    """Choose final/checkpoint XYZ matching final n_atoms in CSV."""
    candidates = [
        (run_dir / "grafeno_cutoff_final.xyz", "final.xyz"),
        (run_dir / "grafeno_cutoff_checkpoint.xyz", "checkpoint.xyz"),
    ]

    for p, label in candidates:
        if p.exists() and xyz_n(p) == expected_n:
            return p, label

    existing = [
        (p, label, xyz_n(p))
        for p, label in candidates
        if p.exists() and xyz_n(p) is not None
    ]

    if not existing:
        raise FileNotFoundError(
            f"No final/checkpoint XYZ found in {run_dir}"
        )

    p, label, n = min(existing, key=lambda x: abs(x[2] - expected_n))
    warnings.warn(
        f"{run_dir}: no XYZ matches CSV final N={expected_n}; "
        f"using {label} with N={n}. Check provenance before publication."
    )
    return p, label + " [COUNT MISMATCH]"


# =============================================================================
# LOAD RUNS
# =============================================================================


def load_run(facet, backend):
    d = backend_dir(facet, backend)

    if not d.exists():
        raise FileNotFoundError(
            f"Missing backend directory for {FACET_LABEL[facet]} / "
            f"{BACKEND_LABEL[backend]}: {d}"
        )

    csvfile = first_existing(d, ["grafeno_cutoff.csv"])
    df = pd.read_csv(csvfile)

    if len(df) == 0:
        raise RuntimeError(f"Empty CSV: {csvfile}")

    if "n_initial" in df:
        n_initial = int(df["n_initial"].iloc[0])
    else:
        raise RuntimeError(f"Column n_initial not found in {csvfile}")

    if "n_atoms" not in df:
        raise RuntimeError(f"Column n_atoms not found in {csvfile}")

    expected_final = int(df["n_atoms"].iloc[-1])
    final_path, final_source = consistent_structure(d, expected_final)
    symf, xyzf = read_xyz(final_path)

    # Initial structure: prefer backend-local file, then GPAW/root file.
    initial_path = first_existing(
        d,
        ["grafeno_cutoff_initial.xyz"],
        required=False,
    )
    if initial_path is None:
        initial_path = first_existing(
            SIROOT / facet,
            ["grafeno_cutoff_initial.xyz"],
            required=False,
        )

    if initial_path is not None:
        sym0, xyz0 = read_xyz(initial_path)
    else:
        # Safe fallback for visualization only: use first n_initial final atoms.
        sym0 = symf[:n_initial].copy()
        xyz0 = xyzf[:n_initial].copy()

    return {
        "facet": facet,
        "backend": backend,
        "dir": d,
        "df": df,
        "n_initial": n_initial,
        "sym0": sym0,
        "xyz0": xyz0,
        "symf": symf,
        "xyzf": xyzf,
        "final_path": final_path,
        "final_source": final_source,
    }


# =============================================================================
# METRIC HELPERS
# =============================================================================


def require_metric(df, metric, where=""):
    if metric not in df.columns:
        raise RuntimeError(
            f"Required metric '{metric}' not found in {where}.\n"
            f"Available columns: {list(df.columns)}"
        )
    return pd.to_numeric(df[metric], errors="coerce")


def final_value(run, metric):
    s = require_metric(run["df"], metric, str(run["dir"])).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.iloc[-1])


def mean_value(run, metric):
    s = require_metric(run["df"], metric, str(run["dir"])).dropna()
    if len(s) == 0:
        return np.nan
    return float(s.mean())


# =============================================================================
# 3D STRUCTURE RENDERING
# =============================================================================


def bond_pairs(xyz, cutoff=SI_BOND_CUTOFF_A):
    xyz = np.asarray(xyz, float)
    pairs = []
    for i in range(len(xyz) - 1):
        delta = xyz[i+1:] - xyz[i]
        d2 = np.einsum("ij,ij->i", delta, delta)
        js = np.where((d2 > 1e-8) & (d2 <= cutoff**2))[0]
        for j0 in js:
            pairs.append((i, i + 1 + int(j0)))
    return pairs


def equalize_3d_box(ax, xyz):
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    center = 0.5 * (mins + maxs)
    span = max(float(np.max(maxs - mins)), 1.0)
    radius = 0.53 * span

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_si111_structure(ax, run):
    """
    Real final Si(111) structure for one backend.

    Gray: atoms belonging to the initial seed.
    Backend color: atoms incorporated by the growth trajectory.
    """
    n0 = run["n_initial"]
    xyzf = np.asarray(run["xyzf"], float)

    if len(xyzf) < n0:
        raise ValueError(
            f"{run['dir']}: final structure shorter than n_initial={n0}"
        )

    zfront0 = float(np.max(run["xyz0"][:, 2]))

    initial_mask_global = np.arange(len(xyzf)) < n0
    added_mask_global = ~initial_mask_global

    display_mask = (
        added_mask_global
        | (
            initial_mask_global
            & (xyzf[:, 2] >= zfront0 - DISPLAY_INITIAL_DEPTH_A)
        )
    )

    indices = np.where(display_mask)[0]
    shown = xyzf[indices].copy()
    shown[:, 2] -= zfront0

    local_of_global = {int(g): int(i) for i, g in enumerate(indices)}

    for i, j in bond_pairs(xyzf):
        if i not in local_of_global or j not in local_of_global:
            continue
        li = local_of_global[i]
        lj = local_of_global[j]
        p1 = shown[li]
        p2 = shown[lj]
        grown = (i >= n0) or (j >= n0)
        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            color=BOND_GROWN if grown else BOND_INITIAL,
            lw=1.0 if grown else 0.65,
            alpha=0.65 if grown else 0.38,
            zorder=1,
        )

    local_initial = np.array([g < n0 for g in indices], dtype=bool)
    local_added = ~local_initial

    if np.any(local_initial):
        ax.scatter(
            shown[local_initial, 0],
            shown[local_initial, 1],
            shown[local_initial, 2],
            s=48,
            c=INITIAL,
            edgecolors=INITIAL_EDGE,
            linewidths=0.45,
            alpha=0.85,
            depthshade=True,
            zorder=3,
        )

    if np.any(local_added):
        ax.scatter(
            shown[local_added, 0],
            shown[local_added, 1],
            shown[local_added, 2],
            s=72,
            c=BACKEND_COLOR[run["backend"]],
            edgecolors=WHITE,
            linewidths=0.8,
            alpha=1.0,
            depthshade=True,
            zorder=5,
        )

    ax.text2D(
        0.5, 1.015,
        BACKEND_LABEL[run["backend"]],
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=13,
        fontweight="bold",
        color=DARK,
        clip_on=False,
    )

    ax.text2D(
        0.98, 0.04,
        rf"$N_{{\rm add}}={int(run['df']['n_added'].iloc[-1])}$",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=DARK,
    )

    ax.view_init(elev=20, azim=-58)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    equalize_3d_box(ax, shown)
    ax.set_axis_off()


# =============================================================================
# PANEL (b): FINAL METRIC HEATMAPS
# =============================================================================


def draw_metric_heatmap(ax, data, title, show_ylabels=False):
    """data shape = facets x backends."""
    im = ax.imshow(
        data,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if np.isfinite(value) and value < 0.55 else DARK
            ax.text(
                j, i,
                f"{value:.2f}" if np.isfinite(value) else "--",
                ha="center", va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(np.arange(len(BACKENDS)))
    ax.set_xticklabels([BACKEND_LABEL[b] for b in BACKENDS], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(FACETS)))

    if show_ylabels:
        ax.set_yticklabels([FACET_LABEL[f] for f in FACETS], fontweight="bold")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    ax.set_title(title, fontweight="bold", fontsize=11.2, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(direction="out")
    return im


# =============================================================================
# MAIN
# =============================================================================


def main():
    if not SIROOT.exists():
        raise FileNotFoundError(f"Cannot find silicon directory: {SIROOT}")

    # -------------------------------------------------------------------------
    # Load all 9 trajectories.
    # -------------------------------------------------------------------------
    runs = {
        (facet, backend): load_run(facet, backend)
        for facet in FACETS
        for backend in BACKENDS
    }

    # -------------------------------------------------------------------------
    # Figure canvas
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(16.4, 11.2), facecolor="white")

    outer = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.06, right=0.985,
        bottom=0.075, top=0.95,
        wspace=0.23, hspace=0.30,
    )

    # =========================================================================
    # PANEL (a): REAL Si(111) STRUCTURES
    # =========================================================================
    sub_a = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[0, 0],
        wspace=-0.03,
    )

    axa = []
    for i, backend in enumerate(BACKENDS):
        ax = fig.add_subplot(sub_a[0, i], projection="3d")
        axa.append(ax)
        plot_si111_structure(ax, runs[("111", backend)])

    panel(axa[0], "a", x=-0.17, y=1.11)
    axa[1].text2D(
        0.5, 1.11,
        "Same Si(111) growth problem, interchangeable energy backend",
        transform=axa[1].transAxes,
        ha="center", va="bottom",
        fontsize=13.5,
        fontweight="bold",
        color=DARK,
        clip_on=False,
    )

    handles_a = [
        Line2D([0], [0], marker="o", ls="", ms=7,
               mfc=INITIAL, mec=INITIAL_EDGE, label="initial Si"),
        Line2D([0], [0], marker="o", ls="", ms=7,
               mfc="#5B7EA5", mec=WHITE, label="incorporated Si"),
    ]
    axa[1].legend(
        handles=handles_a,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.07),
        ncol=2,
        fontsize=8.3,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    # =========================================================================
    # PANEL (b): FINAL STRUCTURAL METRICS
    # =========================================================================
    host_b = fig.add_subplot(outer[0, 1])
    host_b.axis("off")
    panel(host_b, "b", x=-0.10, y=1.08)
    host_b.text(
        0.5, 1.035,
        "Final structural response across facets and backends",
        transform=host_b.transAxes,
        ha="center", va="bottom",
        fontsize=14.5,
        fontweight="bold",
        color=DARK,
    )

    sub_b = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[0, 1],
        wspace=0.22,
    )

    metric_specs = [
        ("template_growth_occupancy_fraction", r"Template\noccupancy"),
        ("fraction_added_CN4", r"Added-Si\n$CN=4$ fraction"),
        ("growth_front_mean_fraction_of_template", r"Normalized\nfront advance"),
    ]

    for k, (metric, title) in enumerate(metric_specs):
        ax = fig.add_subplot(sub_b[0, k])
        matrix = np.array([
            [final_value(runs[(facet, backend)], metric) for backend in BACKENDS]
            for facet in FACETS
        ], float)
        draw_metric_heatmap(
            ax,
            matrix,
            title,
            show_ylabels=(k == 0),
        )

    # =========================================================================
    # PANEL (c): COORDINATION EVOLUTION
    # =========================================================================
    host_c = fig.add_subplot(outer[1, 0])
    host_c.axis("off")
    panel(host_c, "c", x=-0.10, y=1.08)
    host_c.text(
        0.5, 1.035,
        "Tetrahedral ordering during growth",
        transform=host_c.transAxes,
        ha="center", va="bottom",
        fontsize=14.5,
        fontweight="bold",
        color=DARK,
    )

    sub_c = gridspec.GridSpecFromSubplotSpec(
        1, 3,
        subplot_spec=outer[1, 0],
        wspace=0.22,
    )

    for i, facet in enumerate(FACETS):
        ax = fig.add_subplot(sub_c[0, i])

        for backend in BACKENDS:
            run = runs[(facet, backend)]
            y = require_metric(
                run["df"],
                "fraction_added_CN4",
                str(run["dir"]),
            ).to_numpy(float)
            x = np.linspace(0, 1, len(y))
            ax.plot(
                x, y,
                lw=2.0,
                color=BACKEND_COLOR[backend],
                label=BACKEND_LABEL[backend],
            )

        ax.axhline(1.0, color=GRAY, lw=1.0, ls="--", alpha=0.6)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(0, 1)
        ax.set_title(FACET_LABEL[facet], fontweight="bold", fontsize=11.5)
        ax.set_xlabel("Growth progress", fontweight="bold", fontsize=10.5)

        if i == 0:
            ax.set_ylabel(r"Added Si with $CN=4$", fontweight="bold", fontsize=10.5)
            ax.legend(frameon=False, loc="lower right", fontsize=8.3)
        else:
            ax.set_yticklabels([])

        style(ax)

    # =========================================================================
    # PANEL (d): CANDIDATE SPACE VS SELECTION SHARPNESS
    # =========================================================================
    ax = fig.add_subplot(outer[1, 1])
    panel(ax, "d")

    for backend in BACKENDS:
        for facet in FACETS:
            run = runs[(facet, backend)]
            x = mean_value(run, "n_tested_candidates")
            y = mean_value(run, "chosen_boltzmann_probability")

            ax.scatter(
                x, y,
                s=105,
                marker=FACET_MARKER[facet],
                c=BACKEND_COLOR[backend],
                edgecolors=WHITE,
                linewidths=0.9,
                zorder=5,
            )

    ax.set_xlabel(
        "Mean energetically tested candidates / accepted event",
        fontweight="bold",
    )
    ax.set_ylabel(
        "Mean probability of selected candidate",
        fontweight="bold",
    )
    ax.set_ylim(0, 1.02)
    ax.set_title(
        "Restricted branching of pristine Si growth",
        fontweight="bold",
        pad=9,
    )

    # Backend legend
    backend_handles = [
        Line2D(
            [0], [0], marker="o", ls="", ms=8,
            mfc=BACKEND_COLOR[b], mec=WHITE,
            label=BACKEND_LABEL[b],
        )
        for b in BACKENDS
    ]

    facet_handles = [
        Line2D(
            [0], [0], marker=FACET_MARKER[f], ls="", ms=7,
            mfc=GRAY, mec=WHITE,
            label=FACET_LABEL[f],
        )
        for f in FACETS
    ]

    leg1 = ax.legend(
        handles=backend_handles,
        frameon=False,
        loc="lower left",
        title="Backend",
        title_fontsize=9,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=facet_handles,
        frameon=False,
        loc="lower right",
        title="Facet",
        title_fontsize=9,
    )

    style(ax)

    # =========================================================================
    # SUMMARY CSV
    # =========================================================================
    rows = []

    for facet in FACETS:
        for backend in BACKENDS:
            run = runs[(facet, backend)]
            df = run["df"]

            row = {
                "facet": facet,
                "backend": BACKEND_LABEL[backend],
                "directory": str(run["dir"]),
                "n_initial": run["n_initial"],
                "n_final": int(df["n_atoms"].iloc[-1]),
                "n_added": int(df["n_added"].iloc[-1]) if "n_added" in df else np.nan,
                "template_growth_occupancy_fraction": final_value(
                    run, "template_growth_occupancy_fraction"
                ),
                "fraction_added_CN4": final_value(
                    run, "fraction_added_CN4"
                ),
                "growth_front_mean_fraction_of_template": final_value(
                    run, "growth_front_mean_fraction_of_template"
                ),
                "mean_n_tested_candidates": mean_value(
                    run, "n_tested_candidates"
                ),
                "mean_chosen_boltzmann_probability": mean_value(
                    run, "chosen_boltzmann_probability"
                ),
                "structure_source": run["final_source"],
            }

            if "roughness_A" in df:
                row["final_roughness_A"] = float(
                    pd.to_numeric(df["roughness_A"], errors="coerce").iloc[-1]
                )

            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(HERE / f"{OUT}_summary.csv", index=False)

    # =========================================================================
    # SAVE
    # =========================================================================
    out = HERE / OUT

    fig.savefig(
        out.with_suffix(".pdf"),
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        out.with_suffix(".svg"),
        bbox_inches="tight",
        pad_inches=0.08,
    )

    fig.savefig(
        out.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )

    # =========================================================================
    # TERMINAL REPORT
    # =========================================================================
    print("\nGenerated:")
    print(" ", out.with_suffix(".pdf"))
    print(" ", out.with_suffix(".svg"))
    print(" ", out.with_suffix(".png"))
    print(" ", HERE / f"{OUT}_summary.csv")

    print("\nRuns used:")
    for facet in FACETS:
        for backend in BACKENDS:
            run = runs[(facet, backend)]
            print(
                f"  {FACET_LABEL[facet]:7s}  "
                f"{BACKEND_LABEL[backend]:6s}  "
                f"N={len(run['xyzf']):4d}  "
                f"source={run['final_source']}  "
                f"dir={run['dir']}"
            )

    print(
        "\nNOTE: these are complete same-seed trajectories. "
        "They demonstrate backend portability and structural consistency, "
        "not a strict same-candidate energy benchmark."
    )

    plt.show()


if __name__ == "__main__":
    main()
