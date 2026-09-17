#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Results Fig. 3 — Structural generality across 2D, 3D and layered carbon growth

Place at:
    PAPER/figure3/figure3_structural_generality.py

Reads:
    PAPER/graphene/
    PAPER/diamond/
    PAPER/graphitte/

Panels
------
(a) Real final structures: graphene, diamond, graphite.
    Initial atoms = gray; incorporated atoms = material-specific color.

(b) Distribution of unrelaxed insertion energies evaluated by CHGNet.

(c) Growth-front advancement versus number of incorporated atoms.

(d) Local coordination fidelity:
        <CN_added>/CN_ideal
    with CN_ideal=3 for graphene/graphite and CN_ideal=4 for diamond.

If seed_01/, seed_02/, ... subdirectories containing the corresponding CSV
are later added, panels (b)--(d) automatically use all detected seeds.

Outputs
-------
Figure3_Structural_Generality.pdf
Figure3_Structural_Generality.svg
Figure3_Structural_Generality.png
Figure3_Structural_Generality_summary.csv

Dependencies
------------
numpy
pandas
matplotlib

ASE is NOT required.
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
OUT = "Figure3_Structural_Generality"


# =============================================================================
# SYSTEM DEFINITIONS
# =============================================================================

SYSTEMS = ("graphene", "diamond", "graphite")

ROOT = {
    "graphene": PAPER / "graphene",
    "diamond":  PAPER / "diamond",
    "graphite": PAPER / "graphitte",
}

LABEL = {
    "graphene": "Graphene",
    "diamond":  "Diamond",
    "graphite": "Graphite",
}

CLASS = {
    "graphene": "2D covalent",
    "diamond":  "3D tetrahedral",
    "graphite": "layered",
}

CSV_NAME = {
    "graphene": "grafeno_cutoff.csv",
    "diamond":  "diamond_cut.csv",
    "graphite": "grafeno_cutoff.csv",
}

LOG_NAME = {
    "graphene": "grafeno_cutoff.log",
    "diamond":  "diamond.log",
    "graphite": "grafeno_cutoff.log",
}

INITIAL_XYZ = {
    "graphene": "grafeno_cutoff_initial.xyz",
    "diamond":  "diamond_cut_initial.xyz",
    "graphite": "grafeno_cutoff_initial.xyz",
}

FINAL_XYZ = {
    "graphene": "grafeno_cutoff_final.xyz",
    "diamond":  "diamond_cut_final.xyz",
    "graphite": "grafeno_cutoff_final.xyz",
}

CHECKPOINT_XYZ = {
    "graphene": "grafeno_cutoff_checkpoint.xyz",
    "diamond":  "diamond_cut_checkpoint.xyz",
    "graphite": "grafeno_cutoff_checkpoint.xyz",
}

# Ideal local coordination of the grown carbon network.
CN_IDEAL = {
    "graphene": 3.0,
    "diamond":  4.0,
    "graphite": 3.0,
}

# Growth direction from the archived production runs.
GROWTH_AXIS = {
    "graphene": "x",
    "diamond":  "z",
    "graphite": "z",
}

# Material-specific colors, consistent across the figure.
COLOR = {
    "graphene": "#2878B5",
    "diamond":  "#D97706",
    "graphite": "#26958B",
}

DARK = "#17212B"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
INITIAL = "#ADB4BC"
INITIAL_EDGE = "#737B84"
BOND_INITIAL = "#A7ADB4"
BOND_GROWN = "#4B5563"
WHITE = "#FFFFFF"

# C-C first-neighbor visualization cutoff. Interlayer graphite atoms are not
# connected because their separation is much larger than this value.
C_BOND_CUTOFF_A = 1.85

# How much of the original crystal is displayed behind the growth front.
INITIAL_DEPTH_A = {
    "graphene": 5.0,
    "diamond":  5.5,
    "graphite": 13.0,
}

# Camera settings chosen to show dimensionality clearly.
VIEW = {
    "graphene": (19, -62),
    "diamond":  (20, -53),
    "graphite": (20, -58),
}


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


def panel(ax, letter, x=-0.12, y=1.09):
    if hasattr(ax, "text2D"):
        ax.text2D(
            x, y, letter, transform=ax.transAxes,
            fontsize=23, fontweight="bold",
            ha="left", va="top", color="black", clip_on=False,
        )
    else:
        ax.text(
            x, y, letter, transform=ax.transAxes,
            fontsize=23, fontweight="bold",
            ha="left", va="top", color="black", clip_on=False,
        )


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


# =============================================================================
# XYZ / LOG HELPERS
# =============================================================================

def read_xyz(path):
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ: {path}")
    n = int(lines[0].strip())
    sym, xyz = [], []
    for line in lines[2:2+n]:
        f = line.split()
        if len(f) < 4:
            continue
        sym.append(f[0])
        xyz.append([float(f[1]), float(f[2]), float(f[3])])
    if len(xyz) != n:
        raise ValueError(f"XYZ mismatch in {path}: expected {n}, read {len(xyz)}")
    return np.asarray(sym, object), np.asarray(xyz, float)


def xyz_n(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return int(path.read_text(errors="replace").splitlines()[0].strip())
    except Exception:
        return None


def parse_log(log_text):
    backend = None
    m = re.search(r"Backend\s*=\s*([^;\n]+)", log_text)
    if m:
        backend = m.group(1).strip()

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

    mg = re.search(r"growth_axis\s*=\s*([xyz])", log_text)
    growth_axis = mg.group(1) if mg else None

    return {
        "backend": backend,
        "p": p,
        "q": q,
        "gaps": gaps,
        "growth_axis": growth_axis,
    }


def consistent_structure(system, expected_n):
    d = ROOT[system]
    candidates = [
        (d / FINAL_XYZ[system], "final.xyz"),
        (d / CHECKPOINT_XYZ[system], "checkpoint.xyz"),
    ]
    for p, name in candidates:
        if p.exists() and xyz_n(p) == expected_n:
            return p, name

    existing = [(p, name, xyz_n(p)) for p, name in candidates if p.exists()]
    existing = [x for x in existing if x[2] is not None]
    if not existing:
        raise FileNotFoundError(f"No final/checkpoint structure for {system}")

    p, name, n = min(existing, key=lambda x: abs(x[2] - expected_n))
    warnings.warn(
        f"{system}: no XYZ matches CSV N={expected_n}; using {name} with N={n}."
    )
    return p, name + " [COUNT MISMATCH]"


def load_base(system):
    d = ROOT[system]
    df = pd.read_csv(d / CSV_NAME[system])
    log = (d / LOG_NAME[system]).read_text(errors="replace")
    _, xyz0 = read_xyz(d / INITIAL_XYZ[system])

    expected_n = int(df["n_atoms"].iloc[-1])
    final_path, final_source = consistent_structure(system, expected_n)
    _, xyzf = read_xyz(final_path)

    return {
        "df": df,
        "log": log,
        "meta": parse_log(log),
        "xyz0": xyz0,
        "xyzf": xyzf,
        "final_path": final_path,
        "final_source": final_source,
    }


# =============================================================================
# SEED DISCOVERY
# =============================================================================

def seed_csvs(system):
    d = ROOT[system]
    target = CSV_NAME[system]
    found = []
    for pat in (
        f"seed*/{target}", f"seed_*/{target}",
        f"run*/{target}", f"rep*/{target}",
    ):
        found += list(d.glob(pat))
    found = sorted(set(p.resolve() for p in found))
    if len(found) >= 2:
        return found
    return [(d / target).resolve()]


def seed_dfs(system):
    return [pd.read_csv(p) for p in seed_csvs(system)]


def ensemble_curve(dfs, metric, ngrid=240, normalize_x=False):
    curves = []
    if normalize_x:
        xg = np.linspace(0.0, 1.0, ngrid)
    else:
        max_n = max(float(df["n_added"].iloc[-1]) for df in dfs)
        xg = np.linspace(1.0, max_n, ngrid)

    for df in dfs:
        if metric not in df:
            continue
        y = np.asarray(df[metric], float)
        if normalize_x:
            x = np.linspace(0.0, 1.0, len(y))
        else:
            x = np.asarray(df["n_added"], float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            continue
        # For raw n_added, do not extrapolate beyond the run endpoint.
        if normalize_x:
            yi = np.interp(xg, x[ok], y[ok])
        else:
            yi = np.interp(xg, x[ok], y[ok], left=np.nan, right=np.nan)
        curves.append(yi)

    arr = np.vstack(curves)
    mean = np.nanmean(arr, axis=0)
    if len(arr) > 1:
        sd = np.nanstd(arr, axis=0, ddof=1)
    else:
        sd = np.zeros_like(mean)
    return xg, mean, sd


# =============================================================================
# 3D BALL-AND-STICK RENDERING
# =============================================================================

def bond_pairs(xyz, cutoff=C_BOND_CUTOFF_A):
    xyz = np.asarray(xyz, float)
    pairs = []
    for i in range(len(xyz)-1):
        d = xyz[i+1:] - xyz[i]
        d2 = np.einsum("ij,ij->i", d, d)
        js = np.where((d2 > 1e-8) & (d2 <= cutoff**2))[0]
        for j0 in js:
            pairs.append((i, i+1+int(j0)))
    return pairs


def equalize_3d(ax, xyz, margin=0.6):
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    centers = 0.5*(mins+maxs)
    span = max(float(np.max(maxs-mins)), 1.0)
    r = 0.55*span + margin
    ax.set_xlim(centers[0]-r, centers[0]+r)
    ax.set_ylim(centers[1]-r, centers[1]+r)
    ax.set_zlim(centers[2]-r, centers[2]+r)
    try:
        ax.set_box_aspect((1,1,1))
    except Exception:
        pass


def plot_structure_3d(ax, system, run):
    n0 = len(run["xyz0"])
    xyzf = np.asarray(run["xyzf"], float)
    g = {"x":0, "y":1, "z":2}[GROWTH_AXIS[system]]

    if len(xyzf) < n0:
        raise ValueError(f"{system}: final structure has fewer atoms than initial structure")

    front0 = float(np.max(run["xyz0"][:,g]))
    initial_mask = np.arange(len(xyzf)) < n0
    added_mask = ~initial_mask

    # Keep all incorporated atoms, and only the near-front portion of the seed.
    display_mask = added_mask | (
        initial_mask & (xyzf[:,g] >= front0 - INITIAL_DEPTH_A[system])
    )
    idx = np.where(display_mask)[0]
    shown = xyzf[idx].copy()
    shown[:,g] -= front0

    local_of_global = {int(gi):int(li) for li,gi in enumerate(idx)}

    # Bonds
    for i,j in bond_pairs(xyzf):
        if i not in local_of_global or j not in local_of_global:
            continue
        li, lj = local_of_global[i], local_of_global[j]
        p1, p2 = shown[li], shown[lj]
        grown = (i >= n0) or (j >= n0)
        ax.plot(
            [p1[0],p2[0]], [p1[1],p2[1]], [p1[2],p2[2]],
            color=BOND_GROWN if grown else BOND_INITIAL,
            lw=1.0 if grown else 0.7,
            alpha=0.65 if grown else 0.38,
            zorder=1,
        )

    local_initial = np.array([gi < n0 for gi in idx], bool)
    local_added = ~local_initial

    if np.any(local_initial):
        ax.scatter(
            shown[local_initial,0], shown[local_initial,1], shown[local_initial,2],
            s=42, c=INITIAL, edgecolors=INITIAL_EDGE, linewidths=0.45,
            alpha=0.82, depthshade=True, zorder=3,
        )
    if np.any(local_added):
        ax.scatter(
            shown[local_added,0], shown[local_added,1], shown[local_added,2],
            s=56, c=COLOR[system], edgecolors=WHITE, linewidths=0.65,
            alpha=1.0, depthshade=True, zorder=5,
        )

    elev, azim = VIEW[system]
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass
    equalize_3d(ax, shown)
    ax.set_axis_off()

    meta = run["meta"]
    gap_txt = "/".join(f"{g0:.2f}" for g0 in meta["gaps"]) + " Å" if meta["gaps"] else "n/a"

    ax.text2D(
        0.50, 1.02, LABEL[system], transform=ax.transAxes,
        ha="center", va="bottom", fontsize=13.2, fontweight="bold", color=DARK,
    )
    ax.text2D(
        0.03, 0.96,
        f"{CLASS[system]}\n" + rf"$p={meta['p']}$, $\Delta h={gap_txt}$",
        transform=ax.transAxes,
        ha="left", va="top", fontsize=8.7, fontweight="bold", color=DARK,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=LIGHT, alpha=0.88),
    )
    ax.text2D(
        0.97, 0.04,
        rf"$N_0={n0}$" + "\n" + rf"$N_{{\rm add}}={int(run['df']['n_added'].iloc[-1])}$",
        transform=ax.transAxes,
        ha="right", va="bottom", fontsize=8.8, fontweight="bold", color=DARK,
    )


# =============================================================================
# PANEL B — INSERTION-ENERGY DISTRIBUTIONS
# =============================================================================

def plot_energy_distributions(ax, ensemble):
    panel(ax, "b")

    values = []
    for system in SYSTEMS:
        arr = np.concatenate([
            np.asarray(df["e_ads_unrelaxed_eV"], float)
            for df in ensemble[system]
        ])
        arr = arr[np.isfinite(arr)]
        values.append(arr)

    vp = ax.violinplot(
        values,
        positions=np.arange(3),
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, system in zip(vp["bodies"], SYSTEMS):
        body.set_facecolor(COLOR[system])
        body.set_edgecolor(COLOR[system])
        body.set_alpha(0.22)
        body.set_linewidth(1.3)

    for i, (system, arr) in enumerate(zip(SYSTEMS, values)):
        q1, med, q3 = np.percentile(arr, [25,50,75])
        ax.plot([i,i], [q1,q3], color=COLOR[system], lw=5.0, solid_capstyle="round", zorder=5)
        ax.scatter([i], [med], s=58, c=COLOR[system], ec="white", lw=0.8, zorder=6)

    ax.axhline(0, color=LIGHT, lw=1.2)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([LABEL[s] for s in SYSTEMS], fontweight="bold")
    ax.set_ylabel(r"Unrelaxed insertion energy, $\Delta E_{\rm ins}$ (eV)", fontweight="bold")
    ax.set_title("Energetic landscape sampled by CHGNet", fontweight="bold", pad=9)
    style(ax)


# =============================================================================
# PANEL C — FRONT ADVANCEMENT
# =============================================================================

def plot_front_advance(ax, ensemble):
    panel(ax, "c")

    for system in SYSTEMS:
        x, mean, sd = ensemble_curve(ensemble[system], "growth_front_mean_A", normalize_x=False)
        ok = np.isfinite(mean)
        ax.plot(x[ok], mean[ok], color=COLOR[system], lw=2.4, label=LABEL[system])
        if len(ensemble[system]) > 1:
            ax.fill_between(
                x[ok], mean[ok]-sd[ok], mean[ok]+sd[ok],
                color=COLOR[system], alpha=0.16, lw=0,
            )

    ax.axhline(0, color=LIGHT, lw=1.1)
    ax.set_xlabel("Incorporated C atoms", fontweight="bold")
    ax.set_ylabel("Mean growth-front advance (Å)", fontweight="bold")
    ax.set_title("Growth across distinct structural dimensionalities", fontweight="bold", pad=9)
    ax.legend(frameon=False, loc="upper left")
    style(ax)


# =============================================================================
# PANEL D — NORMALIZED COORDINATION FIDELITY
# =============================================================================

def plot_coordination_fidelity(ax, ensemble):
    panel(ax, "d")

    for system in SYSTEMS:
        # Build a normalized metric for every seed dataframe.
        transformed = []
        for df in ensemble[system]:
            tmp = df.copy()
            tmp["coord_fidelity"] = (
                np.asarray(tmp["coordination_added_mean"], float) / CN_IDEAL[system]
            )
            transformed.append(tmp)

        x, mean, sd = ensemble_curve(transformed, "coord_fidelity", normalize_x=True)
        ax.plot(x, mean, color=COLOR[system], lw=2.4, label=LABEL[system])
        if len(transformed) > 1:
            ax.fill_between(
                x,
                np.clip(mean-sd, 0, None),
                mean+sd,
                color=COLOR[system], alpha=0.16, lw=0,
            )

    ax.axhline(1.0, color=GRAY, lw=1.3, ls="--", alpha=0.75)
    ax.text(
        0.98, 0.98, "ideal local coordination",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.2, fontweight="bold", color=GRAY,
    )
    ax.set_xlabel("Normalized accepted-growth progress", fontweight="bold")
    ax.set_ylabel(r"Coordination fidelity, $\langle CN_{\rm add}\rangle/CN_{\rm ideal}$", fontweight="bold")
    ax.set_title("Recovery of material-specific local order", fontweight="bold", pad=9)
    ax.set_ylim(0.0, 1.08)
    style(ax)


# =============================================================================
# MAIN
# =============================================================================

def main():
    for system in SYSTEMS:
        if not ROOT[system].exists():
            raise FileNotFoundError(f"Missing directory: {ROOT[system]}")

    base = {s: load_base(s) for s in SYSTEMS}
    ensemble = {s: seed_dfs(s) for s in SYSTEMS}

    fig = plt.figure(figsize=(16.2, 11.2), facecolor="white")
    outer = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.065, right=0.985, bottom=0.08, top=0.94,
        wspace=0.23, hspace=0.30,
    )

    # -------------------------------------------------------------------------
    # (a) Three real grown structures
    # -------------------------------------------------------------------------
    sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0,0], wspace=-0.04)
    axs3d = []
    for i, system in enumerate(SYSTEMS):
        ax3 = fig.add_subplot(sub[0,i], projection="3d")
        axs3d.append(ax3)
        plot_structure_3d(ax3, system, base[system])

    panel(axs3d[0], "a", x=-0.18, y=1.11)
    axs3d[1].text2D(
        0.50, 1.12,
        "One growth engine across distinct carbon architectures",
        transform=axs3d[1].transAxes,
        ha="center", va="bottom",
        fontsize=15.3, fontweight="bold", color=DARK, clip_on=False,
    )

    axs3d[1].legend(
        handles=[
            Line2D([0],[0], marker="o", ls="", ms=7.5, mfc=INITIAL, mec=INITIAL_EDGE, label="initial C"),
            Line2D([0],[0], marker="o", ls="", ms=7.5, mfc="#2C8A9A", mec="white", label="incorporated C"),
        ],
        loc="lower center", bbox_to_anchor=(0.50,-0.075), ncol=2,
        frameon=False, fontsize=8.3, handletextpad=0.35, columnspacing=0.9,
    )

    # -------------------------------------------------------------------------
    # (b)-(d)
    # -------------------------------------------------------------------------
    axb = fig.add_subplot(outer[0,1])
    axc = fig.add_subplot(outer[1,0])
    axd = fig.add_subplot(outer[1,1])

    plot_energy_distributions(axb, ensemble)
    plot_front_advance(axc, ensemble)
    plot_coordination_fidelity(axd, ensemble)

    # -------------------------------------------------------------------------
    # Summary CSV
    # -------------------------------------------------------------------------
    rows = []
    for system in SYSTEMS:
        df = base[system]["df"]
        energies = np.asarray(df["e_ads_unrelaxed_eV"], float)
        rows.append({
            "system": system,
            "class": CLASS[system],
            "backend": base[system]["meta"]["backend"],
            "growth_axis": base[system]["meta"]["growth_axis"],
            "motif_period_p": base[system]["meta"]["p"],
            "gap_period_q": base[system]["meta"]["q"],
            "gaps_A": ";".join(f"{x:.6f}" for x in base[system]["meta"]["gaps"]),
            "n_initial": len(base[system]["xyz0"]),
            "n_final": len(base[system]["xyzf"]),
            "n_added": int(df["n_added"].iloc[-1]),
            "structure_source": base[system]["final_source"],
            "mean_insertion_energy_eV": float(np.nanmean(energies)),
            "median_insertion_energy_eV": float(np.nanmedian(energies)),
            "final_front_advance_A": float(df["growth_front_mean_A"].iloc[-1]),
            "final_coordination_added_mean": float(df["coordination_added_mean"].iloc[-1]),
            "ideal_coordination": CN_IDEAL[system],
            "final_coordination_fidelity": float(df["coordination_added_mean"].iloc[-1] / CN_IDEAL[system]),
            "n_seed_runs": len(ensemble[system]),
        })

    pd.DataFrame(rows).to_csv(HERE / f"{OUT}_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out = HERE / OUT
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)

    print("\nGenerated:")
    print(" ", out.with_suffix(".pdf"))
    print(" ", out.with_suffix(".svg"))
    print(" ", out.with_suffix(".png"))
    print(" ", HERE / f"{OUT}_summary.csv")

    print("\nRuns used:")
    for system in SYSTEMS:
        df = base[system]["df"]
        print(
            f"  {LABEL[system]}: N0={len(base[system]['xyz0'])}, "
            f"Nadd={int(df['n_added'].iloc[-1])}, Nfinal={len(base[system]['xyzf'])}, "
            f"source={base[system]['final_source']}, seeds={len(ensemble[system])}"
        )

    plt.show()


if __name__ == "__main__":
    main()
