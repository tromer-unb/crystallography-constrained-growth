#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 5 — Programmable disorder during crystallography-constrained graphene growth

Place at:
    PAPER/figure5/figure5.py

Reads the REAL production runs:
    PAPER/graphene/                       pristine
    PAPER/graphene/vac10p/                10% vacancy probability
    PAPER/graphene/vac10p_10pN/           10% vacancy + 10% N probability

Panel (a)
---------
Real final atomistic structures from grafeno_cutoff_final.xyz. The initial
seed is shown in light gray, incorporated C in blue, incorporated N in orange,
and frozen vacancy sites are shown at the ACTUAL recorded template positions
from grafeno_cutoff_vacancies.csv. C-C/C-N bonds are inferred from the final
atomic coordinates; this is not a schematic lattice.

Panel (b)
---------
Realized event fractions from the stochastic trajectories, compared with the
nominal programmed probabilities.

Panel (c)
---------
Unrelaxed insertion-energy distributions for actual incorporated atoms. Vacancy
events are omitted because no atom is inserted and their recorded insertion
energy is identically zero by construction.

Panel (d)
---------
Final structural response: template-site occupancy, local coordination fidelity
< CN_added >/3, and normalized growth-front advance.

Optional multiple seeds
-----------------------
If subdirectories such as seed_01/, seed_02/, ... are later created inside each
case directory, panels (b) and (d) automatically report ensemble means and
standard deviations. Panel (a) always displays the representative base run.

Outputs
-------
Figure5_Programmable_Defects.pdf
Figure5_Programmable_Defects.svg
Figure5_Programmable_Defects.png
Figure5_Programmable_Defects_summary.csv

Dependencies
------------
numpy
pandas
matplotlib

ASE is not required.
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
GRAPHENE = PAPER / "graphene"
OUT = "Figure5_Programmable_Defects"

CASES = ("pristine", "vac10p", "vac10p_10pN")
CASE_DIR = {
    "pristine": GRAPHENE,
    "vac10p": GRAPHENE / "vac10p",
    "vac10p_10pN": GRAPHENE / "vac10p_10pN",
}
CASE_LABEL = {
    "pristine": "Pristine",
    "vac10p": "10% vacancy",
    "vac10p_10pN": "10% vacancy + 10% N",
}

# Nominal event probabilities used in the production scripts.
NOMINAL = {
    "pristine": {"C": 1.00, "N": 0.00, "vac": 0.00},
    "vac10p": {"C": 0.90, "N": 0.00, "vac": 0.10},
    "vac10p_10pN": {"C": 0.80, "N": 0.10, "vac": 0.10},
}


# =============================================================================
# COLORS / STYLE
# =============================================================================

DARK = "#17212B"
GRAY = "#6B7280"
LIGHT = "#D1D5DB"
WHITE = "#FFFFFF"

INITIAL_C = "#B8BFC6"
ADDED_C = "#2878B5"
N_COLOR = "#D97706"
VAC_COLOR = "#C94F3D"
BOND_COLOR = "#9098A1"

METRIC_OCC = "#477BA8"
METRIC_CN = "#4D8B57"
METRIC_FRONT = "#C87519"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11.0,
    "ytick.labelsize": 11.0,
    "legend.fontsize": 9.3,
    "axes.linewidth": 1.25,
    "xtick.major.width": 1.15,
    "ytick.major.width": 1.15,
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


def panel(ax, letter, x=-0.13, y=1.09):
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


# =============================================================================
# XYZ / RUN LOADERS
# =============================================================================


def read_xyz(path):
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 2:
        raise ValueError(f"Invalid XYZ: {path}")
    n = int(lines[0].strip())
    symbols, xyz = [], []
    for line in lines[2:2+n]:
        f = line.split()
        if len(f) < 4:
            continue
        symbols.append(f[0])
        xyz.append([float(f[1]), float(f[2]), float(f[3])])
    if len(xyz) != n:
        raise ValueError(f"XYZ mismatch in {path}: expected {n}, read {len(xyz)}")
    return np.asarray(symbols, object), np.asarray(xyz, float)


def xyz_n(path):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return int(path.read_text(errors="replace").splitlines()[0].strip())
    except Exception:
        return None


def consistent_final(run_dir, expected_n):
    candidates = [
        run_dir / "grafeno_cutoff_final.xyz",
        run_dir / "grafeno_cutoff_checkpoint.xyz",
    ]
    for p in candidates:
        if p.exists() and xyz_n(p) == expected_n:
            return p
    existing = [(p, xyz_n(p)) for p in candidates if p.exists() and xyz_n(p) is not None]
    if not existing:
        raise FileNotFoundError(f"No final/checkpoint XYZ in {run_dir}")
    p, n = min(existing, key=lambda x: abs(x[1] - expected_n))
    warnings.warn(
        f"{run_dir}: no XYZ exactly matches CSV final N={expected_n}; "
        f"using {p.name} with N={n}."
    )
    return p


def load_run(case):
    d = CASE_DIR[case]
    csvfile = d / "grafeno_cutoff.csv"
    initial = d / "grafeno_cutoff_initial.xyz"
    vacancies = d / "grafeno_cutoff_vacancies.csv"
    if not csvfile.exists():
        raise FileNotFoundError(csvfile)
    if not initial.exists():
        raise FileNotFoundError(initial)

    df = pd.read_csv(csvfile)
    sym0, xyz0 = read_xyz(initial)
    n0 = int(df["n_initial"].iloc[0]) if "n_initial" in df else len(xyz0)
    expected_n = int(df["n_atoms"].iloc[-1])
    final_path = consistent_final(d, expected_n)
    symf, xyzf = read_xyz(final_path)

    if vacancies.exists():
        vac = pd.read_csv(vacancies)
    else:
        vac = pd.DataFrame(columns=["x_A", "y_A", "z_A"])

    return {
        "case": case,
        "dir": d,
        "df": df,
        "sym0": sym0,
        "xyz0": xyz0,
        "n0": n0,
        "symf": symf,
        "xyzf": xyzf,
        "vac": vac,
        "final_path": final_path,
    }


# =============================================================================
# OPTIONAL MULTI-SEED DISCOVERY
# =============================================================================


def seed_csvs(case):
    d = CASE_DIR[case]
    found = []
    for pat in (
        "seed*/grafeno_cutoff.csv",
        "seed_*/grafeno_cutoff.csv",
        "run*/grafeno_cutoff.csv",
        "rep*/grafeno_cutoff.csv",
    ):
        found.extend(d.glob(pat))
    found = sorted(set(p.resolve() for p in found))
    if len(found) >= 2:
        return found
    return [(d / "grafeno_cutoff.csv").resolve()]


def seed_dfs(case):
    return [pd.read_csv(p) for p in seed_csvs(case)]


# =============================================================================
# REAL STRUCTURAL RENDERING
# =============================================================================


def bond_pairs_xy(xyz, cutoff=1.75):
    """
    Infer real C-C / C-N nearest-neighbor bonds from final coordinates.
    Graphene calculations here are non-periodic in x/y, so direct distances
    are appropriate for this visualization.
    """
    xyz = np.asarray(xyz, float)
    pairs = []
    n = len(xyz)
    c2 = cutoff * cutoff
    for i in range(n - 1):
        delta = xyz[i+1:] - xyz[i]
        d2 = np.einsum("ij,ij->i", delta, delta)
        js = np.where((d2 > 1e-8) & (d2 <= c2))[0]
        for j0 in js:
            pairs.append((i, i + 1 + int(j0)))
    return pairs


def plot_real_defect_structure(ax, run, label, show_ylabel=True):
    """
    Plot the actual final simulated graphene structure in top view (x-y).

    To make the defect region readable, the figure starts a few Å before the
    initial growth front and includes the entire grown region. This is a crop
    of the REAL final simulation, not a reconstructed ideal lattice.
    """
    n0 = run["n0"]
    symf = np.asarray(run["symf"])
    xyzf = np.asarray(run["xyzf"], float)
    xyz0 = np.asarray(run["xyz0"], float)

    if len(xyzf) < n0:
        raise ValueError(f"{label}: final structure shorter than initial structure")

    initial_front = float(np.max(xyz0[:, 0]))
    crop_min = initial_front - 4.0
    crop_max = float(np.max(xyzf[:, 0])) + 0.8

    mask = (xyzf[:, 0] >= crop_min) & (xyzf[:, 0] <= crop_max)
    idx = np.where(mask)[0]
    local = {int(g): i for i, g in enumerate(idx)}
    shown_xyz = xyzf[idx]
    shown_sym = symf[idx]

    # Bonds from actual final coordinates.
    for i, j in bond_pairs_xy(xyzf, cutoff=1.75):
        if i not in local or j not in local:
            continue
        # Only graphene-like C/N bonds are rendered.
        if symf[i] not in ("C", "N") or symf[j] not in ("C", "N"):
            continue
        p1, p2 = xyzf[i], xyzf[j]
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]],
            color=BOND_COLOR,
            lw=0.52,
            alpha=0.52,
            zorder=1,
        )

    is_initial = idx < n0
    is_added = ~is_initial
    is_n = shown_sym == "N"
    is_c = shown_sym == "C"

    # Initial seed atoms visible near the interface.
    m = is_initial & is_c
    if np.any(m):
        ax.scatter(
            shown_xyz[m, 0], shown_xyz[m, 1],
            s=18, c=INITIAL_C, edgecolors=WHITE, linewidths=0.20,
            alpha=0.82, zorder=3,
        )

    # Added carbon.
    m = is_added & is_c
    if np.any(m):
        ax.scatter(
            shown_xyz[m, 0], shown_xyz[m, 1],
            s=22, c=ADDED_C, edgecolors=WHITE, linewidths=0.28,
            alpha=0.98, zorder=4,
        )

    # Added nitrogen, if present.
    m = is_added & is_n
    if np.any(m):
        ax.scatter(
            shown_xyz[m, 0], shown_xyz[m, 1],
            s=31, c=N_COLOR, edgecolors=WHITE, linewidths=0.40,
            alpha=1.0, zorder=6,
        )

    # Actual frozen-vacancy coordinates recorded by the simulation.
    vac = run["vac"]
    if len(vac):
        v = vac[(vac["x_A"] >= crop_min) & (vac["x_A"] <= crop_max)]
        if len(v):
            ax.scatter(
                v["x_A"], v["y_A"],
                s=44, facecolors="none", edgecolors=VAC_COLOR,
                linewidths=1.15, marker="o", zorder=8,
            )
            ax.scatter(
                v["x_A"], v["y_A"],
                s=18, c=VAC_COLOR, marker="x", linewidths=1.0, zorder=9,
            )

    # Initial growth front only; no extra annotation clutter.
    ax.axvline(initial_front, color=GRAY, lw=0.9, ls="--", alpha=0.65, zorder=2)

    ax.set_xlim(crop_min, crop_max)
    yall = shown_xyz[:, 1]
    ax.set_ylim(float(yall.min()) - 0.7, float(yall.max()) + 0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(label, fontweight="bold", pad=7)
    ax.set_xlabel(r"$x$ (Å)", fontweight="bold")
    if show_ylabel:
        ax.set_ylabel(r"$y$ (Å)", fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    style(ax)


# =============================================================================
# STATISTICS
# =============================================================================


def realized_fractions(df):
    n = len(df)
    species = df["deposited_species"].astype(str).str.strip()
    return {
        "C": float((species == "C").sum() / n),
        "N": float((species == "N").sum() / n),
        "vac": float((species == "vac").sum() / n),
    }


def ensemble_event_stats(case):
    vals = {"C": [], "N": [], "vac": []}
    for df in seed_dfs(case):
        f = realized_fractions(df)
        for k in vals:
            vals[k].append(f[k])
    result = {}
    for k, arr in vals.items():
        a = np.asarray(arr, float)
        result[k] = (
            float(a.mean()),
            float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        )
    return result


def final_metric_stats(case, metric, transform=None):
    vals = []
    for df in seed_dfs(case):
        if metric not in df:
            continue
        v = float(df[metric].iloc[-1])
        if transform is not None:
            v = transform(v)
        vals.append(v)
    a = np.asarray(vals, float)
    if len(a) == 0:
        return np.nan, np.nan
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


# =============================================================================
# PANEL (b): REALIZED COMPOSITION / DEFECT FRACTIONS
# =============================================================================


def plot_event_fractions(ax):
    panel(ax, "b")

    x = np.arange(len(CASES), dtype=float)
    bottoms = np.zeros(len(CASES))

    species_order = ("C", "N", "vac")
    colors = {"C": ADDED_C, "N": N_COLOR, "vac": VAC_COLOR}
    labels = {"C": "C incorporation", "N": "N substitution", "vac": "vacancy"}

    means_by_sp = {sp: [] for sp in species_order}
    sd_by_sp = {sp: [] for sp in species_order}

    for case in CASES:
        st = ensemble_event_stats(case)
        for sp in species_order:
            means_by_sp[sp].append(st[sp][0])
            sd_by_sp[sp].append(st[sp][1])

    for sp in species_order:
        vals = np.asarray(means_by_sp[sp])
        ax.bar(
            x, vals, bottom=bottoms, width=0.58,
            color=colors[sp], edgecolor=WHITE, linewidth=0.8,
            label=labels[sp], zorder=3,
        )
        # Label only substantial fractions to avoid clutter.
        for xx, val, bot in zip(x, vals, bottoms):
            if val >= 0.055:
                ax.text(
                    xx, bot + 0.5*val, f"{100*val:.1f}%",
                    ha="center", va="center", fontsize=8.8,
                    fontweight="bold", color=WHITE,
                )
        bottoms += vals

    ax.set_ylim(0, 1.04)
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABEL[c] for c in CASES], fontweight="bold", rotation=8)
    ax.set_ylabel("Fraction of growth events", fontweight="bold")
    ax.set_title("Realized defect chemistry", fontweight="bold", pad=8)
    ax.legend(frameon=False, loc="lower left", fontsize=8.7)
    style(ax)


# =============================================================================
# PANEL (c): INSERTION-ENERGY DISTRIBUTIONS
# =============================================================================


def plot_energy_distributions(ax, runs):
    panel(ax, "c")

    groups = []
    labels = []
    colors = []

    selections = [
        ("pristine", "C", "Pristine\nC", ADDED_C),
        ("vac10p", "C", "Vacancy\nC", "#4D89A8"),
        ("vac10p_10pN", "C", "Vac.+N\nC", "#6B8FA6"),
        ("vac10p_10pN", "N", "Vac.+N\nN", N_COLOR),
    ]

    for case, species, lab, color in selections:
        df = runs[case]["df"]
        m = df["deposited_species"].astype(str).str.strip() == species
        values = pd.to_numeric(df.loc[m, "e_ads_unrelaxed_eV"], errors="coerce").dropna().to_numpy(float)
        groups.append(values)
        labels.append(lab)
        colors.append(color)

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.58,
        showfliers=False,
        medianprops=dict(color=DARK, linewidth=1.5),
        whiskerprops=dict(color=GRAY, linewidth=1.0),
        capprops=dict(color=GRAY, linewidth=1.0),
        boxprops=dict(linewidth=1.0),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
        patch.set_edgecolor(color)

    for i, values in enumerate(groups, start=1):
        # Minimal sample-size label.
        ax.text(
            i, ax.get_ylim()[0] if False else np.nanmedian(values),
            "", ha="center"
        )

    ax.axhline(0, color=LIGHT, lw=1.0)
    ax.set_xticks(np.arange(1, len(labels)+1))
    ax.set_xticklabels(labels, rotation=0, ha="center", fontweight="bold", fontsize=9.0)
    ax.set_ylabel(r"Unrelaxed insertion energy, $\Delta E_{\rm ins}$ (eV)", fontweight="bold")
    ax.set_title("Insertion-energy distributions", fontweight="bold", pad=8)
    style(ax)


# =============================================================================
# PANEL (d): FINAL STRUCTURAL RESPONSE
# =============================================================================


def plot_structural_metrics(ax):
    panel(ax, "d")

    metrics = [
        (
            "template_growth_occupancy_fraction",
            "Template occupancy",
            METRIC_OCC,
            None,
        ),
        (
            "coordination_added_mean",
            r"Coordination fidelity $\langle CN\rangle/3$",
            METRIC_CN,
            lambda x: x / 3.0,
        ),
        (
            "growth_front_mean_fraction_of_template",
            "Normalized front advance",
            METRIC_FRONT,
            None,
        ),
    ]

    x = np.arange(len(CASES), dtype=float)
    w = 0.22
    offsets = (-w, 0.0, w)

    for j, (metric, label, color, transform) in enumerate(metrics):
        means, sds = [], []
        for case in CASES:
            m, s = final_metric_stats(case, metric, transform=transform)
            means.append(m)
            sds.append(s)
        means = np.asarray(means, float)
        sds = np.asarray(sds, float)

        ax.bar(
            x + offsets[j], means,
            width=w*0.90,
            yerr=sds if np.any(sds > 0) else None,
            capsize=3,
            color=color,
            alpha=0.78,
            edgecolor=color,
            linewidth=1.0,
            label=label,
            zorder=3,
        )

    ax.axhline(1.0, color=GRAY, lw=1.0, ls="--", alpha=0.65)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels([CASE_LABEL[c] for c in CASES], fontweight="bold", rotation=8)
    ax.set_ylabel("Normalized structural metric", fontweight="bold")
    ax.set_title("Structural response to defects", fontweight="bold", pad=8)
    ax.legend(frameon=False, loc="lower left", fontsize=8.2)
    style(ax)


# =============================================================================
# MAIN
# =============================================================================


def main():
    if not GRAPHENE.exists():
        raise FileNotFoundError(f"Cannot find {GRAPHENE}. Put this script inside PAPER/figure5/.")

    for case in CASES:
        if not CASE_DIR[case].exists():
            raise FileNotFoundError(CASE_DIR[case])

    runs = {case: load_run(case) for case in CASES}

    # Wide top structural panel + three compact quantitative panels below.
    fig = plt.figure(figsize=(17.6, 8.9), facecolor="white")
    outer = gridspec.GridSpec(
        2, 3, figure=fig,
        height_ratios=[0.78, 1.00],
        left=0.055, right=0.985,
        bottom=0.09, top=0.94,
        wspace=0.25, hspace=0.36,
    )

    # -------------------------------------------------------------------------
    # (a) REAL SIMULATED STRUCTURES
    # -------------------------------------------------------------------------
    sub = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, :], wspace=0.10)
    axs_a = [fig.add_subplot(sub[0, i]) for i in range(3)]

    for i, case in enumerate(CASES):
        plot_real_defect_structure(
            axs_a[i], runs[case], CASE_LABEL[case], show_ylabel=(i == 0)
        )

    panel(axs_a[0], "a", x=-0.12, y=1.12)
    axs_a[1].text(
        0.5, 1.12,
        "Real simulated graphene structures",
        transform=axs_a[1].transAxes,
        ha="center", va="bottom",
        fontsize=15.2, fontweight="bold", color=DARK,
        clip_on=False,
    )

    handles = [
        Line2D([0], [0], marker="o", ls="", ms=7, mfc=INITIAL_C, mec=WHITE, label="initial C"),
        Line2D([0], [0], marker="o", ls="", ms=7, mfc=ADDED_C, mec=WHITE, label="incorporated C"),
        Line2D([0], [0], marker="o", ls="", ms=7, mfc=N_COLOR, mec=WHITE, label="substitutional N"),
        Line2D([0], [0], marker="o", ls="", ms=8, mfc="none", mec=VAC_COLOR, label="frozen vacancy"),
    ]
    fig.legend(
        handles=handles, frameon=False, ncol=4,
        loc="upper center", bbox_to_anchor=(0.50, 0.585),
        fontsize=8.9, columnspacing=1.2, handletextpad=0.4,
    )

    # -------------------------------------------------------------------------
    # (b)-(d)
    # -------------------------------------------------------------------------
    ax_b = fig.add_subplot(outer[1, 0])
    ax_c = fig.add_subplot(outer[1, 1])
    ax_d = fig.add_subplot(outer[1, 2])

    plot_event_fractions(ax_b)
    plot_energy_distributions(ax_c, runs)
    plot_structural_metrics(ax_d)

    # -------------------------------------------------------------------------
    # Summary CSV
    # -------------------------------------------------------------------------
    rows = []
    for case in CASES:
        df = runs[case]["df"]
        frac = realized_fractions(df)
        row = {
            "case": case,
            "label": CASE_LABEL[case],
            "n_events": len(df),
            "n_initial": int(df["n_initial"].iloc[0]),
            "n_added": int(df["n_added"].iloc[-1]),
            "n_vacancies": int(df["n_vacancies"].iloc[-1]),
            "realized_C_fraction": frac["C"],
            "realized_N_fraction": frac["N"],
            "realized_vacancy_fraction": frac["vac"],
            "nominal_C_fraction": NOMINAL[case]["C"],
            "nominal_N_fraction": NOMINAL[case]["N"],
            "nominal_vacancy_fraction": NOMINAL[case]["vac"],
            "template_occupancy": float(df["template_growth_occupancy_fraction"].iloc[-1]),
            "frozen_vacancy_fraction_of_template": float(df["frozen_vacancy_fraction_of_template"].iloc[-1]),
            "coordination_added_mean": float(df["coordination_added_mean"].iloc[-1]),
            "coordination_fidelity_CN3": float(df["coordination_added_mean"].iloc[-1]) / 3.0,
            "front_fraction_of_template": float(df["growth_front_mean_fraction_of_template"].iloc[-1]),
            "roughness_A": float(df["roughness_A"].iloc[-1]),
            "structure_file": runs[case]["final_path"].name,
        }
        rows.append(row)

    pd.DataFrame(rows).to_csv(HERE / f"{OUT}_summary.csv", index=False)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    out = HERE / OUT
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.08)

    print("\nGenerated:")
    for ext in ("pdf", "svg", "png"):
        print(" ", out.with_suffix("." + ext))
    print(" ", HERE / f"{OUT}_summary.csv")

    print("\nRepresentative runs:")
    for case in CASES:
        df = runs[case]["df"]
        frac = realized_fractions(df)
        print(
            f"  {CASE_LABEL[case]}: Nfinal={len(runs[case]['xyzf'])}, "
            f"C={100*frac['C']:.1f}%, N={100*frac['N']:.1f}%, "
            f"vac={100*frac['vac']:.1f}%, vacancies={len(runs[case]['vac'])}"
        )

    print("\nSeed ensembles detected:")
    for case in CASES:
        print(f"  {CASE_LABEL[case]}: n={len(seed_csvs(case))}")

    if all(len(seed_csvs(case)) == 1 for case in CASES):
        print(
            "\nNOTE: the current archive contains one stochastic trajectory per case. "
            "The code will automatically add ensemble error bars when seed_*/ runs are added."
        )

    plt.show()


if __name__ == "__main__":
    main()
