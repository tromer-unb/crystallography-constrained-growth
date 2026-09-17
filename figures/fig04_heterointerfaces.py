#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 4 — Heterostructure and multicomponent growth
REAL atomistic rendering from simulation XYZ files.

Place at:
    PAPER/figure4/figure4.py

Reads the actual final/checkpoint structures in:
    PAPER/SiC/001
    PAPER/SiC/010
    PAPER/SiC/100
    PAPER/MoS2              (current user tree)

For compatibility with the archived PAPER used during development it also
accepts:
    PAPER/Mos2_about_grap

Panel (a): real top-view ball-and-stick renderings of graphene grown laterally
           in +x on three SiC orientations. Initial substrate/graphene and
           incorporated graphene atoms are separated using the actual run.
Panel (b): final SiC growth metrics.
Panel (c): real top-view ball-and-stick rendering of MoS2 grown on graphene.
Panel (d): species-resolved Mo/S insertion-energy distributions.

Outputs:
    Figure4_Heterostructure_Multicomponent.pdf
    Figure4_Heterostructure_Multicomponent.svg
    Figure4_Heterostructure_Multicomponent.png
    Figure4_Heterostructure_Multicomponent_summary.csv

Dependencies:
    numpy, pandas, matplotlib

ASE is not required.
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
SIC_ROOT = PAPER / "SiC"
OUT = "Figure4_Heterostructure_Multicomponent"

SIC_CASES = ("001", "010", "100")
SIC_LABEL = {"001": "SiC(001)", "010": "SiC(010)", "100": "SiC(100)"}
SIC_COLOR = {"001": "#2474A6", "010": "#D47A16", "100": "#258D82"}

# Scientific color palette
DARK = "#17212B"
GRAY = "#66707A"
LIGHT = "#D7DCE1"
WHITE = "#FFFFFF"
SI_COLOR = "#D7DCE2"
SI_EDGE = "#9DA5AE"
SUB_C_COLOR = "#676F78"
SEED_GRAPHENE = "#90D7C9"
GRAPHENE_BOND = "#58616B"

GRAPHENE_BG = "#C3C8CE"
MO_INITIAL = "#A49AE0"
MO_ADDED = "#5A49CB"
S_INITIAL = "#F3D17C"
S_ADDED = "#E39B21"
MOS_BOND = "#626873"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9.5,
    "axes.linewidth": 1.25,
    "xtick.major.width": 1.1,
    "ytick.major.width": 1.1,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def panel(ax, letter, x=-0.08, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=23,
            fontweight="bold", ha="left", va="top", clip_on=False, color="black")


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")
    ax.grid(False)


def scale_bar(ax, length=5.0, label=None, color=DARK, lw=2.4):
    """Add a simple physical scale bar to an atomistic rendering."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    x0 = xmin + 0.055 * (xmax - xmin)
    y0 = ymin + 0.065 * (ymax - ymin)
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=lw,
            solid_capstyle="butt", zorder=30)
    ax.text(x0 + length / 2, y0 + 0.025 * (ymax - ymin),
            label or f"{length:g} Å", ha="center", va="bottom",
            fontsize=8.8, fontweight="bold", color=color, zorder=31)


# =============================================================================
# FILE READERS
# =============================================================================
def read_extxyz(path):
    """Read species, Cartesian coordinates, cell, and pbc from extxyz."""
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ: {path}")
    n = int(lines[0].strip())
    comment = lines[1]

    m = re.search(r'Lattice="([^"]+)"', comment)
    if m:
        vals = [float(x) for x in m.group(1).split()]
        cell = np.asarray(vals, float).reshape(3, 3)
    else:
        cell = np.eye(3)

    pm = re.search(r'pbc="([^"]+)"', comment)
    if pm:
        pbc = np.asarray([t.upper().startswith("T") for t in pm.group(1).split()], bool)
    else:
        pbc = np.zeros(3, bool)

    symbols, xyz = [], []
    for line in lines[2:2+n]:
        f = line.split()
        if len(f) < 4:
            continue
        symbols.append(f[0])
        xyz.append([float(f[1]), float(f[2]), float(f[3])])

    if len(xyz) != n:
        raise ValueError(f"XYZ mismatch in {path}: expected {n}, read {len(xyz)}")
    return np.asarray(symbols, object), np.asarray(xyz, float), cell, pbc


def xyz_n(path):
    try:
        return int(Path(path).read_text(errors="replace").splitlines()[0].strip())
    except Exception:
        return None


def choose_final_structure(run_dir, expected_n):
    """Choose final.xyz or checkpoint.xyz whose atom count matches the CSV."""
    run_dir = Path(run_dir)
    candidates = [
        run_dir / "grafeno_cutoff_final.xyz",
        run_dir / "grafeno_cutoff_checkpoint.xyz",
    ]
    for p in candidates:
        if p.exists() and xyz_n(p) == expected_n:
            return p
    existing = [p for p in candidates if p.exists() and xyz_n(p) is not None]
    if not existing:
        raise FileNotFoundError(f"No final/checkpoint XYZ in {run_dir}")
    p = min(existing, key=lambda q: abs(xyz_n(q) - expected_n))
    warnings.warn(f"{run_dir}: no structure has N={expected_n}; using {p.name} with N={xyz_n(p)}")
    return p


def parse_restrict_z(log_text):
    patterns = [
        r"Restrição cristalográfica de crescimento:.*?Z>=([0-9.Ee+\-]+)",
        r"restrict_fractional=\([^,]*,[^,]*,\s*([0-9.Ee+\-]+)\)",
    ]
    for pat in patterns:
        m = re.search(pat, log_text, re.S)
        if m:
            return float(m.group(1))
    return None


def fractional_coordinates(xyz, cell):
    return np.linalg.solve(np.asarray(cell, float).T, np.asarray(xyz, float).T).T


# =============================================================================
# GEOMETRIC/BOND HELPERS
# =============================================================================
def pair_list(xyz, cutoff, species=None, allowed_pairs=None, min_dist=0.6):
    """Return real-space bonds inferred directly from the final coordinates."""
    xyz = np.asarray(xyz, float)
    n = len(xyz)
    out = []
    for i in range(n - 1):
        d = xyz[i+1:] - xyz[i]
        dist = np.linalg.norm(d, axis=1)
        js = np.where((dist >= min_dist) & (dist <= cutoff))[0]
        for j0 in js:
            j = i + 1 + int(j0)
            if species is not None and allowed_pairs is not None:
                pair = frozenset((str(species[i]), str(species[j])))
                if pair not in allowed_pairs:
                    continue
            out.append((i, j))
    return out


def draw_bonds_xy(ax, xyz, pairs, color, lw=1.0, alpha=0.7, zorder=2):
    for i, j in pairs:
        ax.plot([xyz[i, 0], xyz[j, 0]], [xyz[i, 1], xyz[j, 1]],
                color=color, lw=lw, alpha=alpha,
                solid_capstyle="round", zorder=zorder)


# =============================================================================
# LOAD RUNS
# =============================================================================
def load_sic_case(ori):
    d = SIC_ROOT / ori
    df = pd.read_csv(d / "grafeno_cutoff.csv")
    log = (d / "grafeno_cutoff.log").read_text(errors="replace")
    sy0, x0, cell0, pbc0 = read_extxyz(d / "grafeno_cutoff_initial.xyz")
    n0 = int(df["n_initial"].iloc[0]) if "n_initial" in df else len(x0)
    nf = int(df["n_atoms"].iloc[-1])
    fp = choose_final_structure(d, nf)
    syf, xf, cellf, pbcf = read_extxyz(fp)

    rz = parse_restrict_z(log)
    if rz is None:
        raise RuntimeError(f"Could not identify restrict_Z in {d/'grafeno_cutoff.log'}")

    frac0 = fractional_coordinates(x0, cell0)
    seed_idx = np.where((sy0 == "C") & (frac0[:, 2] >= rz))[0]
    if len(seed_idx) == 0:
        raise RuntimeError(f"No graphene seed atoms found for {ori} using restrict_Z={rz}")

    return dict(dir=d, df=df, log=log, sy0=sy0, x0=x0, cell0=cell0,
                n0=n0, syf=syf, xf=xf, final_path=fp,
                restrict_z=rz, seed_idx=seed_idx)


def detect_mos2_dir():
    # User's current directory is PAPER/MoS2. The second entry keeps the script
    # compatible with the archived PAPER tree used to test the figure.
    candidates = [PAPER / "MoS2", PAPER / "Mos2_about_grap",
                  PAPER / "MoS2_graphene", PAPER / "mos2_graphene"]
    for d in candidates:
        if (d / "grafeno_cutoff.csv").exists() and (d / "grafeno_cutoff_initial.xyz").exists():
            return d
    raise FileNotFoundError("Could not locate MoS2 run. Expected PAPER/MoS2/grafeno_cutoff.csv")


def load_mos2_case():
    d = detect_mos2_dir()
    df = pd.read_csv(d / "grafeno_cutoff.csv")
    sy0, x0, cell0, pbc0 = read_extxyz(d / "grafeno_cutoff_initial.xyz")
    n0 = int(df["n_initial"].iloc[0]) if "n_initial" in df else len(x0)
    nf = int(df["n_atoms"].iloc[-1])
    fp = choose_final_structure(d, nf)
    syf, xf, cellf, pbcf = read_extxyz(fp)
    return dict(dir=d, df=df, sy0=sy0, x0=x0, n0=n0,
                syf=syf, xf=xf, final_path=fp)


# =============================================================================
# PANEL (a): REAL GRAPHENE/SiC STRUCTURES
# =============================================================================
def render_sic_topview(ax, ori, run):
    """Actual final simulation geometry, viewed along z to expose lateral growth."""
    n0 = run["n0"]
    syf = run["syf"]
    xf = run["xf"]
    seed_idx = run["seed_idx"]

    if len(xf) < n0:
        raise RuntimeError(f"{ori}: final structure has fewer atoms than n_initial")

    initial_idx = np.arange(n0)
    seed_set = set(int(i) for i in seed_idx)
    substrate_idx = np.asarray([i for i in initial_idx if i not in seed_set], int)
    added_idx = np.arange(n0, len(xf))
    grown_c_idx = added_idx[syf[added_idx] == "C"]

    # Use the relaxed final positions of the original graphene seed.
    seed_final = xf[seed_idx]
    grown_final = xf[grown_c_idx]
    graphene_idx = np.concatenate((seed_idx, grown_c_idx)).astype(int)
    graphene_xyz = xf[graphene_idx]

    z_graph = float(np.median(seed_final[:, 2]))
    # Show only the top ~5 Å of the actual SiC substrate; the full substrate is
    # still in the simulation, but a transparent background avoids burying the graphene lattice.
    top_sub_idx = substrate_idx[xf[substrate_idx, 2] >= z_graph - 5.0]
    top_si = top_sub_idx[syf[top_sub_idx] == "Si"]
    top_c = top_sub_idx[syf[top_sub_idx] == "C"]

    # Actual C-C network from the final simulation geometry.
    bonds = pair_list(graphene_xyz, cutoff=1.78, min_dist=1.10)
    draw_bonds_xy(ax, graphene_xyz, bonds, GRAPHENE_BOND, lw=0.95, alpha=0.72, zorder=4)

    # Real SiC substrate in the background.
    if len(top_si):
        ax.scatter(xf[top_si, 0], xf[top_si, 1], s=18, c=SI_COLOR,
                   edgecolors=SI_EDGE, linewidths=0.25, alpha=0.35, zorder=1)
    if len(top_c):
        ax.scatter(xf[top_c, 0], xf[top_c, 1], s=18, c=SUB_C_COLOR,
                   edgecolors="none", alpha=0.26, zorder=2)

    # Pre-existing graphene seed.
    ax.scatter(seed_final[:, 0], seed_final[:, 1], s=45, c=SEED_GRAPHENE,
               edgecolors=WHITE, linewidths=0.60, alpha=1.0, zorder=6)

    # Carbon incorporated during the actual simulation.
    if len(grown_final):
        ax.scatter(grown_final[:, 0], grown_final[:, 1], s=48, c=SIC_COLOR[ori],
                   edgecolors=WHITE, linewidths=0.60, alpha=1.0, zorder=7)

    initial_front = float(np.max(seed_final[:, 0]))
    final_front = float(np.max(graphene_xyz[:, 0]))
    ax.axvline(initial_front, color="#56AFA0", lw=1.25, ls="--", alpha=0.8, zorder=3)

    # Growth arrow is anchored to the actual seed and final graphene fronts.
    ymin = float(np.min(graphene_xyz[:, 1])); ymax = float(np.max(graphene_xyz[:, 1]))
    yarrow = ymax + 0.07 * max(ymax - ymin, 1.0)
    ax.annotate("", xy=(final_front, yarrow), xytext=(initial_front, yarrow),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5,
                                mutation_scale=12), zorder=20)
    ax.text((initial_front + final_front) / 2, yarrow + 0.02 * max(ymax-ymin, 1.0),
            r"growth $+x$", ha="center", va="bottom", fontsize=8.7,
            fontweight="bold", color=DARK)

    # Crop around the REAL final graphene ribbon, not the complete simulation box.
    xmin = float(np.min(graphene_xyz[:, 0])); xmax = float(np.max(graphene_xyz[:, 0]))
    xpad = max(0.7, 0.025 * (xmax - xmin))
    ypad = max(0.6, 0.07 * (ymax - ymin))
    ax.set_xlim(xmin - xpad, xmax + xpad)
    ax.set_ylim(ymin - ypad, yarrow + 0.7 * ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    theta = float(run["df"]["template_growth_occupancy_fraction"].iloc[-1])
    ax.text(0.02, 0.98, SIC_LABEL[ori], transform=ax.transAxes,
            ha="left", va="top", fontsize=12.5, fontweight="bold", color=DARK)
    ax.text(0.98, 0.055, rf"$\theta={theta:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, fontweight="bold", color=DARK)
    scale_bar(ax, 5.0)


# =============================================================================
# PANEL (b): SiC METRICS
# =============================================================================
def plot_sic_metrics(ax, runs):
    panel(ax, "b", x=-0.13, y=1.08)
    x = np.arange(3, dtype=float)
    w = 0.28
    occ = np.asarray([float(runs[o]["df"]["template_growth_occupancy_fraction"].iloc[-1]) for o in SIC_CASES])
    front = np.asarray([float(runs[o]["df"]["growth_front_mean_fraction_of_template"].iloc[-1]) for o in SIC_CASES])

    ax.bar(x-w/2, occ, width=w, color="#CFE1F3", edgecolor="#477BA8",
           linewidth=1.3, label="Template occupancy", zorder=3)
    ax.bar(x+w/2, front, width=w, color="#DCEFE7", edgecolor="#4D8B57",
           linewidth=1.3, label="Front advance", zorder=3)
    for xx, yy in zip(x-w/2, occ):
        ax.text(xx, yy+0.018, f"{yy:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=DARK)
    for xx, yy in zip(x+w/2, front):
        ax.text(xx, yy+0.018, f"{yy:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=DARK)
    ax.axhline(1.0, color=GRAY, lw=1.0, ls="--", alpha=0.65)
    ax.set_ylim(0, 1.10)
    ax.set_xticks(x)
    ax.set_xticklabels([SIC_LABEL[o] for o in SIC_CASES], fontweight="bold")
    ax.set_ylabel("Dimensionless growth metric", fontweight="bold")
    ax.set_title("Graphene propagation on SiC", fontweight="bold", pad=9)
    ax.legend(frameon=False, loc="lower center", ncol=2, fontsize=9)
    style(ax)


# =============================================================================
# PANEL (c): REAL MoS2/GRAPHENE STRUCTURE
# =============================================================================
def render_mos2_topview(ax, run):
    panel(ax, "c", x=-0.09, y=1.08)
    n0 = run["n0"]
    syf = run["syf"]
    xf = run["xf"]

    old_idx = np.arange(n0)
    add_idx = np.arange(n0, len(xf))
    old_c = old_idx[syf[old_idx] == "C"]
    old_mo = old_idx[syf[old_idx] == "Mo"]
    old_s = old_idx[syf[old_idx] == "S"]
    add_mo = add_idx[syf[add_idx] == "Mo"]
    add_s = add_idx[syf[add_idx] == "S"]

    mos_idx = np.concatenate((old_mo, old_s, add_mo, add_s)).astype(int)
    mos_xyz = xf[mos_idx]
    mos_sy = syf[mos_idx]

    # Real Mo-S bonds from the final trajectory.
    bonds = pair_list(mos_xyz, cutoff=2.75, species=mos_sy,
                      allowed_pairs={frozenset(("Mo", "S"))}, min_dist=1.5)
    draw_bonds_xy(ax, mos_xyz, bonds, MOS_BOND, lw=1.0, alpha=0.67, zorder=3)

    # Graphene support, real final coordinates, kept faint.
    ax.scatter(xf[old_c, 0], xf[old_c, 1], s=15, c=GRAPHENE_BG,
               edgecolors="none", alpha=0.33, zorder=1)

    if len(old_mo):
        ax.scatter(xf[old_mo, 0], xf[old_mo, 1], s=62, c=MO_INITIAL,
                   edgecolors=WHITE, linewidths=0.6, zorder=5, label="initial Mo")
    if len(old_s):
        ax.scatter(xf[old_s, 0], xf[old_s, 1], s=46, c=S_INITIAL,
                   edgecolors=WHITE, linewidths=0.55, zorder=5, label="initial S")
    if len(add_mo):
        ax.scatter(xf[add_mo, 0], xf[add_mo, 1], s=66, c=MO_ADDED,
                   edgecolors=WHITE, linewidths=0.65, zorder=7, label="added Mo")
    if len(add_s):
        ax.scatter(xf[add_s, 0], xf[add_s, 1], s=49, c=S_ADDED,
                   edgecolors=WHITE, linewidths=0.60, zorder=7, label="added S")

    seed_growth = np.concatenate((old_mo, old_s)).astype(int)
    final_growth = mos_idx
    initial_front = float(np.max(xf[seed_growth, 0]))
    final_front = float(np.max(xf[final_growth, 0]))
    ymin = float(np.min(xf[final_growth, 1])); ymax = float(np.max(xf[final_growth, 1]))
    yarrow = ymax + 0.07 * max(ymax-ymin, 1.0)
    ax.axvline(initial_front, color="#8175CC", lw=1.2, ls="--", alpha=0.75, zorder=2)
    ax.annotate("", xy=(final_front, yarrow), xytext=(initial_front, yarrow),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5,
                                mutation_scale=12), zorder=20)
    ax.text((initial_front+final_front)/2, yarrow + 0.02*max(ymax-ymin,1.0),
            r"growth $+x$", ha="center", va="bottom", fontsize=8.7,
            fontweight="bold", color=DARK)

    xmin = float(np.min(xf[final_growth, 0])); xmax = float(np.max(xf[final_growth, 0]))
    xpad = max(0.7, 0.035*(xmax-xmin)); ypad = max(0.7, 0.08*(ymax-ymin))
    ax.set_xlim(xmin-xpad, xmax+xpad)
    ax.set_ylim(ymin-ypad, yarrow+0.8*ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(r"MoS$_2$ lateral growth on graphene", fontweight="bold", pad=8)
    theta = float(run["df"]["template_growth_occupancy_fraction"].iloc[-1])
    ax.text(0.98, 0.055, rf"$\theta={theta:.2f}$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, fontweight="bold", color=DARK)
    scale_bar(ax, 5.0)

    handles = [
        Line2D([0],[0], marker='o', ls='', ms=6.5, mfc=GRAPHENE_BG, mec='none', label='graphene C'),
        Line2D([0],[0], marker='o', ls='', ms=7, mfc=MO_INITIAL, mec=WHITE, label='initial Mo'),
        Line2D([0],[0], marker='o', ls='', ms=6.5, mfc=S_INITIAL, mec=WHITE, label='initial S'),
        Line2D([0],[0], marker='o', ls='', ms=7, mfc=MO_ADDED, mec=WHITE, label='added Mo'),
        Line2D([0],[0], marker='o', ls='', ms=6.5, mfc=S_ADDED, mec=WHITE, label='added S'),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.07), ncol=3, fontsize=7.8,
              handletextpad=0.25, columnspacing=0.65)


# =============================================================================
# PANEL (d): Mo/S ENERGETICS
# =============================================================================
def plot_mos2_energies(ax, run):
    panel(ax, "d", x=-0.13, y=1.08)
    df = run["df"]
    scol = "deposited_species" if "deposited_species" in df else "intended_species"
    ecol = "e_ads_unrelaxed_eV"
    sp = df[scol].astype(str).str.strip()
    e = pd.to_numeric(df[ecol], errors="coerce")
    mo = e[sp.str.lower() == "mo"].dropna().to_numpy(float)
    ss = e[sp.str.lower() == "s"].dropna().to_numpy(float)
    if not len(mo) or not len(ss):
        raise RuntimeError(f"Could not find both Mo and S accepted events in {run['dir']/ 'grafeno_cutoff.csv'}")

    emin = min(float(mo.min()), float(ss.min())); emax = max(float(mo.max()), float(ss.max()))
    bins = np.linspace(emin, emax, 17)
    ax.hist(mo, bins=bins, alpha=0.72, color=MO_ADDED, edgecolor=WHITE,
            linewidth=0.65, label=rf"Mo ($n={len(mo)}$)")
    ax.hist(ss, bins=bins, alpha=0.72, color=S_ADDED, edgecolor=WHITE,
            linewidth=0.65, label=rf"S ($n={len(ss)}$)")
    ax.axvline(np.median(mo), color=MO_ADDED, lw=1.25, ls="--")
    ax.axvline(np.median(ss), color=S_ADDED, lw=1.25, ls="--")
    ax.set_xlabel(r"Unrelaxed insertion energy, $\Delta E_{\rm ins}$ (eV)", fontweight="bold")
    ax.set_ylabel("Accepted events", fontweight="bold")
    ax.set_title(r"Species-resolved MoS$_2$ energetics", fontweight="bold", pad=9)
    ax.legend(frameon=False)
    style(ax)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not SIC_ROOT.exists():
        raise FileNotFoundError(f"Missing {SIC_ROOT}")

    sic = {o: load_sic_case(o) for o in SIC_CASES}
    mos = load_mos2_case()

    # Layout: panel (a) gets the entire top row so the atomistic structures are
    # large enough to read as real structures. Panels b-d share the bottom row.
    fig = plt.figure(figsize=(17.0, 8.9), facecolor="white")
    outer = gridspec.GridSpec(2, 3, figure=fig,
                              height_ratios=[0.64, 1.00],
                              left=0.045, right=0.985,
                              bottom=0.075, top=0.955,
                              wspace=0.28, hspace=0.20)

    top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, :], wspace=0.08)
    axs_top = [fig.add_subplot(top[0, i]) for i in range(3)]
    for ax, ori in zip(axs_top, SIC_CASES):
        render_sic_topview(ax, ori, sic[ori])
    panel(axs_top[0], "a", x=-0.06, y=1.08)
    axs_top[1].text(0.5, 1.075,
                    r"Real simulated graphene structures: lateral propagation on SiC in $+x$",
                    transform=axs_top[1].transAxes, ha="center", va="bottom",
                    fontsize=15.0, fontweight="bold", color=DARK, clip_on=False)

    handles = [
        Line2D([0],[0], marker='o', ls='', ms=7, mfc=SI_COLOR, mec=SI_EDGE, label='SiC: Si'),
        Line2D([0],[0], marker='o', ls='', ms=7, mfc=SUB_C_COLOR, mec='none', label='SiC: C'),
        Line2D([0],[0], marker='o', ls='', ms=7, mfc=SEED_GRAPHENE, mec=WHITE, label='initial graphene'),
        Line2D([0],[0], marker='o', ls='', ms=7, mfc='#2474A6', mec=WHITE, label='incorporated graphene C'),
    ]
    axs_top[1].legend(handles=handles, frameon=False, loc="lower center",
                      bbox_to_anchor=(0.5, -0.095), ncol=4, fontsize=8.5,
                      handletextpad=0.35, columnspacing=0.9)

    axb = fig.add_subplot(outer[1, 0]); plot_sic_metrics(axb, sic)
    axc = fig.add_subplot(outer[1, 1]); render_mos2_topview(axc, mos)
    axd = fig.add_subplot(outer[1, 2]); plot_mos2_energies(axd, mos)

    # Summary table
    rows = []
    for ori in SIC_CASES:
        df = sic[ori]["df"]
        rows.append({
            "case": SIC_LABEL[ori],
            "n_initial": int(df["n_initial"].iloc[0]),
            "n_added": int(df["n_added"].iloc[-1]),
            "template_growth_occupancy_fraction": float(df["template_growth_occupancy_fraction"].iloc[-1]),
            "growth_front_mean_fraction_of_template": float(df["growth_front_mean_fraction_of_template"].iloc[-1]),
            "restrict_Z": sic[ori]["restrict_z"],
            "structure_file": sic[ori]["final_path"].name,
        })
    dfm = mos["df"]
    sp = dfm["deposited_species"].astype(str).str.strip() if "deposited_species" in dfm else dfm["intended_species"].astype(str).str.strip()
    rows.append({
        "case": "MoS2/graphene",
        "n_initial": int(dfm["n_initial"].iloc[0]),
        "n_added": int(dfm["n_added"].iloc[-1]),
        "n_Mo_events": int((sp.str.lower()=="mo").sum()),
        "n_S_events": int((sp.str.lower()=="s").sum()),
        "template_growth_occupancy_fraction": float(dfm["template_growth_occupancy_fraction"].iloc[-1]),
        "growth_front_mean_fraction_of_template": float(dfm["growth_front_mean_fraction_of_template"].iloc[-1]),
        "structure_file": mos["final_path"].name,
    })
    pd.DataFrame(rows).to_csv(HERE / f"{OUT}_summary.csv", index=False)

    out = HERE / OUT
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)

    print("\nGenerated:")
    print(" ", out.with_suffix(".pdf"))
    print(" ", out.with_suffix(".svg"))
    print(" ", out.with_suffix(".png"))
    print(" ", HERE / f"{OUT}_summary.csv")
    print("\nStructures:")
    for ori in SIC_CASES:
        print(f"  {SIC_LABEL[ori]}: {sic[ori]['final_path'].name}, N={len(sic[ori]['xf'])}, seed C={len(sic[ori]['seed_idx'])}, added={int(sic[ori]['df']['n_added'].iloc[-1])}")
    print(f"  MoS2/graphene: {mos['dir']}, {mos['final_path'].name}, N={len(mos['xf'])}")
    plt.show()


if __name__ == "__main__":
    main()
