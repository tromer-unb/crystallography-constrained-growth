#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
growth_parameter_advisor.py
===========================

Automatic geometric advisor for template-constrained atomistic growth.

The program reads an initial structure and a crystallographic template (CIF or
any ASE-readable format), analyzes growth independently along x, y and z, and
suggests geometry-derived parameters for mc_flux-style simulations.

Main tasks
----------
1. Detect columns/sublayers along each Cartesian growth axis.
2. Detect periodic patterns in column spacings (gap period).
3. Detect periodic patterns in the transverse atomic motif (column period).
4. Identify truncated edge columns and oblique/hexagonal cells that are unsafe
   for direct Cartesian edge extrapolation.
5. Estimate first and second neighbour shells.
6. Extrapolate the first future template column and measure its distance and
   ideal connectivity to the current structure.
7. Classify the interface as covalent continuation or detached-layer nucleation.
8. Recommend:
      --pbc
      --template-column-period
      --template-column-tol
      --template-layer-tol
      --template-front-min / --template-front-max
      --template-occupancy-tol
      --template-connect-cutoff
      --template-min-neighbors
      --min-dist
      --cn-cutoff
9. When a separate template is supplied, try to identify a simple fractional
   --restrict_X/Y/Z threshold that separates template-matching atoms from
   same-species substrate atoms.
10. Print a ready-to-paste command-line fragment and optionally save JSON.

The recommendations are STARTING VALUES, not replacements for convergence
checks. Energetic parameters (temperature, chemical potentials, DFT cutoffs,
k-points, relaxation convergence, etc.) cannot be inferred from geometry alone.

Requirements
------------
    pip install ase numpy

Examples
--------
Analyze every axis:

    python growth_parameter_advisor.py \
        --structure diamond_332.cif \
        --template diamond_332.cif

Analyze only z:

    python growth_parameter_advisor.py \
        --structure diamond_332.cif \
        --template diamond_332.cif \
        --axis z

Graphene/SiC heterostructure:

    python growth_parameter_advisor.py \
        --structure SiC_grafeno.cif \
        --template hex_grafeno.cif \
        --species C

Save machine-readable report:

    python growth_parameter_advisor.py \
        --structure structure.cif \
        --template template.cif \
        --json advisor.json

Notes
-----
* The growth algorithm is Cartesian. Therefore a crystallographic direction
  intended to grow in x/y/z should be aligned with that Cartesian axis.
* Lateral growth from an oblique hexagonal cell can yield incomplete Cartesian
  edge columns. The advisor detects this and recommends rectangularization.
* A detached new layer (e.g. graphite stacking normal to the basal plane) is
  intentionally treated differently from covalent edge continuation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from ase import Atom, Atoms
    from ase.io import read
except Exception as exc:  # pragma: no cover - user-facing dependency error
    raise SystemExit(
        "This program requires ASE and NumPy. Install them with:\n"
        "    pip install ase numpy\n"
        f"Original import error: {exc}"
    )


AXIS = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = {0: "x", 1: "y", 2: "z"}


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------

@dataclass
class RestrictionSuggestion:
    axis: str
    threshold: float
    matched_min: float
    unmatched_max: float
    gap: float
    confidence: str = "high"


@dataclass
class AxisReport:
    axis: str
    growth_direction: str
    pbc: Tuple[int, int, int]

    n_structure_atoms: int
    n_template_atoms: int
    n_columns: int
    column_counts: List[int]
    edge_column_count: int
    reference_column_count: int
    edge_completeness: float

    min_column_gap_A: Optional[float]
    median_column_gap_A: Optional[float]
    gap_pattern_A: List[float]
    gap_period: Optional[int]
    gap_period_source: str

    structural_period: Optional[int]
    structural_period_source: str
    structural_period_confidence: float

    first_neighbor_A: Optional[float]
    second_neighbor_A: Optional[float]

    first_future_coord_A: Optional[float]
    first_future_count: int
    first_future_dmin_min_A: Optional[float]
    first_future_dmin_median_A: Optional[float]
    first_future_dmin_max_A: Optional[float]
    future_neighbor_counts: List[int]

    interface_mode: str
    oblique_warning: bool
    truncated_edge_warning: bool
    rectangularization_recommended: bool

    recommended_column_tol_A: Optional[float]
    recommended_layer_tol_A: Optional[float]
    recommended_occupancy_tol_A: Optional[float]
    recommended_min_dist_A: Optional[float]
    recommended_connect_cutoff_A: Optional[float]
    recommended_cn_cutoff_A: Optional[float]
    recommended_min_neighbors: int
    recommended_front_min_A: float
    recommended_front_max_A: Optional[float]
    recommended_mobile_radius_A: Optional[float]

    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Basic numerical helpers
# -----------------------------------------------------------------------------

def round_step(value: float, step: float = 0.05, mode: str = "nearest") -> float:
    if step <= 0:
        return float(value)
    x = float(value) / step
    if mode == "up":
        return float(math.ceil(x - 1e-12) * step)
    if mode == "down":
        return float(math.floor(x + 1e-12) * step)
    return float(round(x) * step)


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 <= 1e-14 or n2 <= 1e-14:
        return float("nan")
    c = np.dot(v1, v2) / (n1 * n2)
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def cell_angles_from_vectors(atoms: Atoms) -> Tuple[float, float, float]:
    a, b, c = np.asarray(atoms.cell.array, dtype=float)
    alpha = angle_deg(b, c)
    beta = angle_deg(a, c)
    gamma = angle_deg(a, b)
    return alpha, beta, gamma


def safe_scaled_positions(atoms: Atoms) -> np.ndarray:
    try:
        return np.asarray(atoms.get_scaled_positions(wrap=False), dtype=float)
    except Exception:
        cell = np.asarray(atoms.cell.array, dtype=float)
        return np.linalg.solve(cell.T, np.asarray(atoms.positions, dtype=float).T).T


def pbc_distances_positions_to_point(
    cell: np.ndarray,
    pbc: Sequence[bool],
    positions: np.ndarray,
    point: np.ndarray,
) -> np.ndarray:
    """Minimum-image distances between many Cartesian positions and one point."""
    positions = np.asarray(positions, dtype=float)
    point = np.asarray(point, dtype=float)
    if len(positions) == 0:
        return np.asarray([], dtype=float)

    try:
        frac_pos = np.linalg.solve(cell.T, positions.T).T
        frac_point = np.linalg.solve(cell.T, point)
        dfrac = frac_pos - frac_point
        for ax in range(3):
            if bool(pbc[ax]):
                dfrac[:, ax] -= np.round(dfrac[:, ax])
        dcart = dfrac @ cell
        return np.linalg.norm(dcart, axis=1)
    except np.linalg.LinAlgError:
        return np.linalg.norm(positions - point, axis=1)


def min_distances_to_structure(
    structure: Atoms,
    points: np.ndarray,
    pbc: Sequence[bool],
    species: Optional[set] = None,
) -> np.ndarray:
    mask = np.ones(len(structure), dtype=bool)
    if species is not None:
        mask = np.asarray([a.symbol in species for a in structure], dtype=bool)
    base_pos = np.asarray(structure.positions[mask], dtype=float)
    cell = np.asarray(structure.cell.array, dtype=float)
    out = []
    for p in np.asarray(points, dtype=float):
        d = pbc_distances_positions_to_point(cell, pbc, base_pos, p)
        out.append(float(np.min(d)) if len(d) else float("inf"))
    return np.asarray(out, dtype=float)


def neighbor_counts_for_points(
    structure: Atoms,
    points: np.ndarray,
    pbc: Sequence[bool],
    cutoff: float,
    species: Optional[set] = None,
) -> List[int]:
    mask = np.ones(len(structure), dtype=bool)
    if species is not None:
        mask = np.asarray([a.symbol in species for a in structure], dtype=bool)
    base_pos = np.asarray(structure.positions[mask], dtype=float)
    cell = np.asarray(structure.cell.array, dtype=float)
    counts = []
    for p in np.asarray(points, dtype=float):
        d = pbc_distances_positions_to_point(cell, pbc, base_pos, p)
        counts.append(int(np.sum((d > 1e-8) & (d <= cutoff))))
    return counts


# -----------------------------------------------------------------------------
# Species selection
# -----------------------------------------------------------------------------

def normalize_species_arg(values: Optional[Sequence[str]]) -> Optional[set]:
    if not values:
        return None
    vals = [str(v) for v in values]
    if any(v.lower() == "all" for v in vals):
        return None
    return set(vals)


def filter_atoms_by_species(atoms: Atoms, species: Optional[set]) -> Atoms:
    if species is None:
        return atoms.copy()
    indices = [i for i, a in enumerate(atoms) if a.symbol in species]
    return atoms[indices]


# -----------------------------------------------------------------------------
# Column detection and periodicity
# -----------------------------------------------------------------------------

def raw_unique_coordinates(coords: Sequence[float], merge_tol: float = 1e-3) -> List[float]:
    vals = sorted(float(x) for x in coords)
    unique: List[List[float]] = []
    for x in vals:
        if not unique or abs(x - float(np.mean(unique[-1]))) > merge_tol:
            unique.append([x])
        else:
            unique[-1].append(x)
    return [float(np.mean(g)) for g in unique]


def estimate_column_tol(atoms: Atoms, g: int) -> Tuple[float, Optional[float], List[float]]:
    unique = raw_unique_coordinates(atoms.positions[:, g], merge_tol=1e-3)
    if len(unique) < 2:
        return 0.20, None, unique
    diffs = np.diff(unique)
    diffs = diffs[diffs > 1e-3]
    if len(diffs) == 0:
        return 0.20, None, unique
    dmin = float(np.min(diffs))
    # Enough to absorb coordinate noise, safely below the nearest distinct plane.
    tol = max(0.03, min(0.20, 0.22 * dmin))
    return float(tol), dmin, unique


def axis_columns(atoms: Atoms, g: int, tol: float) -> List[List[int]]:
    order = np.argsort(np.asarray(atoms.positions[:, g], dtype=float))
    groups: List[List[int]] = []
    current: List[int] = []
    current_mean: Optional[float] = None

    for idx0 in order:
        idx = int(idx0)
        x = float(atoms.positions[idx, g])
        if not current:
            current = [idx]
            current_mean = x
            continue

        assert current_mean is not None
        if abs(x - current_mean) <= tol:
            current.append(idx)
            current_mean = float(np.mean(atoms.positions[current, g]))
        else:
            groups.append(current)
            current = [idx]
            current_mean = x

    if current:
        groups.append(current)
    return groups


def quantized_column_signature(
    atoms: Atoms,
    indices: Sequence[int],
    g: int,
    tol: float = 0.01,
) -> Tuple:
    axes = [ax for ax in range(3) if ax != g]
    qtol = max(float(tol), 1e-6)
    rows = []
    for idx in indices:
        atom = atoms[int(idx)]
        coords = tuple(int(np.rint(float(atom.position[ax]) / qtol)) for ax in axes)
        rows.append((atom.symbol,) + coords)
    rows.sort()
    return tuple(rows)


def minimal_period(
    sequence: Sequence,
    equal: Callable[[object, object], bool],
    max_period: int = 24,
    require_two_repeats: bool = True,
) -> Optional[int]:
    n = len(sequence)
    if n == 0:
        return None
    if n == 1:
        return 1

    limit = min(max_period, n - 1)
    if require_two_repeats:
        limit = min(limit, n // 2)

    for p in range(1, limit + 1):
        if require_two_repeats and n < 2 * p:
            continue
        if all(equal(sequence[i], sequence[i - p]) for i in range(p, n)):
            return p
    return None


def best_structural_period(
    atoms: Atoms,
    groups: Sequence[Sequence[int]],
    g: int,
    signature_tol: float = 0.01,
) -> Tuple[int, str, float, List[Tuple]]:
    signatures = [quantized_column_signature(atoms, grp, g, signature_tol) for grp in groups]
    n = len(signatures)
    if n <= 1:
        return 1, "single-column", 1.0, signatures

    p = minimal_period(signatures, lambda a, b: a == b, max_period=24, require_two_repeats=True)
    if p is not None:
        comparisons = max(1, n - p)
        score = sum(signatures[i] == signatures[i - p] for i in range(p, n)) / comparisons
        return int(p), "auto", float(score), signatures

    # A CIF cell itself is periodic. If only one full stacking sequence is present,
    # using all observed columns is the physically conservative closure period.
    return n, "cell-fallback", 0.5, signatures


def detect_gap_pattern(col_coords: Sequence[float], tol: float) -> Tuple[List[float], int, str]:
    coords = np.asarray(col_coords, dtype=float)
    if len(coords) < 2:
        return [], 0, "none"
    diffs = np.diff(coords)
    if np.any(diffs <= 1e-8):
        return [float(np.median(diffs[diffs > 1e-8]))] if np.any(diffs > 1e-8) else [], 1, "median-fallback"

    gap_tol = max(1e-4, min(0.03, 0.08 * max(tol, 1e-3)))
    p = minimal_period(
        list(map(float, diffs)),
        lambda a, b: abs(float(a) - float(b)) <= gap_tol,
        max_period=24,
        require_two_repeats=False,
    )
    if p is None:
        return [float(np.median(diffs))], 1, "median-fallback"

    values = []
    for phase in range(int(p)):
        phase_values = diffs[phase::int(p)]
        values.append(float(np.mean(phase_values)))
    return values, int(p), "auto"


def representative_column_count(counts: Sequence[int]) -> int:
    if not counts:
        return 0
    c = Counter(int(x) for x in counts)
    max_freq = max(c.values())
    modes = [value for value, freq in c.items() if freq == max_freq]
    # Prefer the larger mode when frequencies tie; truncated edge columns are
    # typically smaller than complete interior columns.
    return int(max(modes))


# -----------------------------------------------------------------------------
# Neighbour shells
# -----------------------------------------------------------------------------

def collect_pair_distances(
    atoms: Atoms,
    pbc: Sequence[bool],
    species: Optional[set],
    max_pairs_atoms: int = 700,
    max_distance: float = 6.0,
) -> np.ndarray:
    """Collect pair distances with MIC. Subsamples deterministically if huge."""
    indices = [i for i, a in enumerate(atoms) if species is None or a.symbol in species]
    if len(indices) < 2:
        return np.asarray([], dtype=float)

    if len(indices) > max_pairs_atoms:
        step = max(1, len(indices) // max_pairs_atoms)
        indices = indices[::step][:max_pairs_atoms]

    positions = np.asarray(atoms.positions[indices], dtype=float)
    cell = np.asarray(atoms.cell.array, dtype=float)
    dvals: List[float] = []

    for i in range(len(positions) - 1):
        d = pbc_distances_positions_to_point(cell, pbc, positions[i + 1 :], positions[i])
        if len(d):
            good = d[(d > 0.35) & (d <= max_distance)]
            dvals.extend(float(x) for x in good)

    return np.asarray(dvals, dtype=float)


def cluster_distance_shells(distances: np.ndarray, shell_tol: float = 0.06) -> List[Tuple[float, int]]:
    if len(distances) == 0:
        return []
    vals = np.sort(np.asarray(distances, dtype=float))
    groups: List[List[float]] = []
    current = [float(vals[0])]
    center = float(vals[0])
    for x0 in vals[1:]:
        x = float(x0)
        # Relative allowance prevents over-splitting mildly distorted shells.
        tol = max(shell_tol, 0.025 * max(center, 1.0))
        if abs(x - center) <= tol:
            current.append(x)
            center = float(np.median(current))
        else:
            groups.append(current)
            current = [x]
            center = x
    groups.append(current)
    return [(float(np.median(g)), len(g)) for g in groups]


def first_two_neighbor_shells(
    atoms: Atoms,
    pbc: Sequence[bool],
    species: Optional[set],
) -> Tuple[Optional[float], Optional[float], List[Tuple[float, int]]]:
    d = collect_pair_distances(atoms, pbc, species)
    shells = cluster_distance_shells(d)
    if not shells:
        return None, None, shells
    d1 = shells[0][0]
    d2 = shells[1][0] if len(shells) > 1 else None
    return float(d1), float(d2) if d2 is not None else None, shells


# -----------------------------------------------------------------------------
# PBC and cell-geometry heuristics
# -----------------------------------------------------------------------------

def infer_axis_periodic_from_fractional_span(atoms: Atoms, ax: int) -> bool:
    """Heuristic only: infer whether an axis looks filled rather than vacuum-like."""
    frac = safe_scaled_positions(atoms)[:, ax]
    frac = frac - np.floor(frac)
    if len(frac) <= 1:
        return False
    vals = np.sort(frac)

    # Largest circular empty interval in fractional coordinate.
    gaps = np.diff(np.r_[vals, vals[0] + 1.0])
    largest_empty = float(np.max(gaps))
    occupied_fraction = 1.0 - largest_empty

    # If atoms cover most of the periodic circle, treat as periodic.
    return bool(occupied_fraction >= 0.60)


def seam_min_distance_for_axis(
    atoms: Atoms,
    ax: int,
    species: Optional[set] = None,
) -> Optional[float]:
    """Return the shortest distance created specifically by wrapping across one cell seam.

    Only pairs whose raw fractional separation along ``ax`` exceeds 0.5 are
    considered. This detects a common failure mode in finite CIFs: atoms may span
    almost the entire cell, causing a naive occupancy heuristic to infer PBC even
    though the two Cartesian edges are separated by an unphysical short gap.
    """
    indices = [
        i for i, atom in enumerate(atoms)
        if species is None or atom.symbol in species
    ]
    if len(indices) < 2:
        return None

    frac = safe_scaled_positions(atoms)[indices]
    cell = np.asarray(atoms.cell.array, dtype=float)
    best = float("inf")

    for i in range(len(indices) - 1):
        raw = frac[i + 1:] - frac[i]
        seam_mask = np.abs(raw[:, ax]) > 0.5
        if not np.any(seam_mask):
            continue

        dfrac = raw[seam_mask].copy()
        dfrac[:, ax] -= np.round(dfrac[:, ax])
        dcart = dfrac @ cell
        d = np.linalg.norm(dcart, axis=1)
        good = d[d > 0.35]
        if len(good):
            best = min(best, float(np.min(good)))

    return None if not np.isfinite(best) else float(best)


def intrinsic_first_neighbor_no_pbc(
    atoms: Atoms,
    species: Optional[set] = None,
) -> Optional[float]:
    """Estimate the physical first-neighbour distance without using cell wrapping."""
    d = collect_pair_distances(
        atoms,
        pbc=(False, False, False),
        species=species,
    )
    shells = cluster_distance_shells(d)
    return float(shells[0][0]) if shells else None


def suggest_pbc_with_diagnostics(
    structure: Atoms,
    growth_axis: int,
    species: Optional[set] = None,
) -> Tuple[Tuple[int, int, int], List[str]]:
    """Suggest PBC and reject seams that create non-physical neighbour distances.

    A transverse axis is considered periodic only if:
      1) atoms span enough of that fractional direction to make periodicity plausible;
      2) the shortest bond created by wrapping across the seam is compatible with
         the intrinsic first-neighbour distance measured without PBC.

    This prevents finite ribbons with small cell margins from being misclassified
    as periodic. A graphene ribbon with a 1.0 Å Y seam but 1.42 Å C--C bonds is a
    representative example.
    """
    d1_intrinsic = intrinsic_first_neighbor_no_pbc(structure, species)
    pbc: List[int] = []
    notes: List[str] = []

    for ax in range(3):
        if ax == growth_axis:
            pbc.append(0)
            continue

        if not infer_axis_periodic_from_fractional_span(structure, ax):
            pbc.append(0)
            continue

        seam = seam_min_distance_for_axis(structure, ax, species)
        if d1_intrinsic is None or seam is None:
            # Insufficient evidence for a safe periodic closure.
            pbc.append(0)
            notes.append(
                f"{AXIS_NAME[ax].upper()} periodicity was not enabled automatically "
                "because the cell seam could not be validated geometrically."
            )
            continue

        ratio = seam / max(d1_intrinsic, 1e-12)

        # A valid crystalline seam should be close to a normal first-neighbour
        # connection. The window is deliberately permissive enough for modest
        # strain but rejects artificial sub-bond-length contacts or vacuum gaps.
        if 0.82 <= ratio <= 1.22:
            pbc.append(1)
        else:
            pbc.append(0)
            notes.append(
                f"{AXIS_NAME[ax].upper()} PBC rejected: seam distance={seam:.3f} Å "
                f"is incompatible with intrinsic d1={d1_intrinsic:.3f} Å "
                f"(ratio={ratio:.2f}). Rebuild the cell if periodicity along "
                f"{AXIS_NAME[ax].upper()} is physically required."
            )

    return tuple(pbc), notes  # type: ignore[return-value]


def suggest_pbc(structure: Atoms, growth_axis: int) -> Tuple[int, int, int]:
    """Backward-compatible wrapper."""
    return suggest_pbc_with_diagnostics(structure, growth_axis, species=None)[0]


def cartesian_growth_axis_is_oblique(atoms: Atoms, g: int, tol: float = 1e-3) -> bool:
    """True if other lattice vectors carry a significant component along growth axis."""
    cell = np.asarray(atoms.cell.array, dtype=float)
    scale = max(np.linalg.norm(cell, axis=1).max(), 1.0)
    for row in range(3):
        if row == g:
            continue
        if abs(float(cell[row, g])) > tol * scale:
            return True
    return False


def is_hexagonal_xy(atoms: Atoms, angle_tol: float = 1.0, ab_tol: float = 0.02) -> bool:
    lengths = atoms.cell.lengths()
    alpha, beta, gamma = atoms.cell.angles()
    a, b, _ = map(float, lengths)
    same_ab = abs(a - b) / max(a, b, 1e-12) <= ab_tol
    return bool(
        same_ab
        and abs(float(alpha) - 90.0) <= angle_tol
        and abs(float(beta) - 90.0) <= angle_tol
        and (abs(float(gamma) - 60.0) <= angle_tol or abs(float(gamma) - 120.0) <= angle_tol)
    )


# -----------------------------------------------------------------------------
# Future-column extrapolation
# -----------------------------------------------------------------------------

def extrapolate_first_future_column(
    template: Atoms,
    groups: Sequence[Sequence[int]],
    g: int,
    structural_period: int,
    gap_pattern: Sequence[float],
    direction: str,
) -> Tuple[np.ndarray, Optional[float], int]:
    if len(groups) < 1 or structural_period < 1 or len(gap_pattern) < 1:
        return np.empty((0, 3), dtype=float), None, -1

    col_coords = [float(np.mean(template.positions[list(grp), g])) for grp in groups]
    nbase = len(groups)
    p = min(max(1, int(structural_period)), nbase)
    q = len(gap_pattern)

    if direction == "plus":
        transition_index = nbase - 1
        gap = float(gap_pattern[transition_index % q])
        coord = float(col_coords[-1] + gap)
        source_index = nbase - p
    elif direction == "minus":
        transition_index = -1
        gap = float(gap_pattern[transition_index % q])
        coord = float(col_coords[0] - gap)
        source_index = (p - 1) % p
    else:
        raise ValueError("direction must be 'plus' or 'minus'")

    src = groups[int(source_index)]
    pts = []
    for idx in src:
        pos = np.asarray(template.positions[int(idx)], dtype=float).copy()
        pos[g] = coord
        pts.append(pos)
    return np.asarray(pts, dtype=float), coord, int(source_index)


# -----------------------------------------------------------------------------
# Restriction inference from structure/template matching
# -----------------------------------------------------------------------------

def match_structure_atoms_to_template(
    structure: Atoms,
    template: Atoms,
    species: Optional[set],
    tol_A: float = 0.20,
) -> np.ndarray:
    """Mark structure atoms coincident with template atoms of the same species."""
    matched = np.zeros(len(structure), dtype=bool)
    cell = np.asarray(structure.cell.array, dtype=float)
    # For a CIF-based structural comparison, use the structure's declared PBC.
    pbc = tuple(bool(x) for x in structure.pbc)
    if not any(pbc):
        pbc = (True, True, True)

    template_by_symbol: Dict[str, List[np.ndarray]] = {}
    for a in template:
        if species is not None and a.symbol not in species:
            continue
        template_by_symbol.setdefault(a.symbol, []).append(np.asarray(a.position, dtype=float))

    for i, atom in enumerate(structure):
        if species is not None and atom.symbol not in species:
            continue
        candidates = template_by_symbol.get(atom.symbol, [])
        if not candidates:
            continue
        d = pbc_distances_positions_to_point(
            cell,
            pbc,
            np.asarray(candidates, dtype=float),
            np.asarray(atom.position, dtype=float),
        )
        if len(d) and float(np.min(d)) <= tol_A:
            matched[i] = True
    return matched


def infer_restrictions(
    structure: Atoms,
    template: Atoms,
    species: Optional[set],
    match_tol_A: float = 0.20,
) -> List[RestrictionSuggestion]:
    if len(structure) == 0 or len(template) == 0:
        return []

    matched = match_structure_atoms_to_template(structure, template, species, tol_A=match_tol_A)
    considered = np.asarray(
        [species is None or atom.symbol in species for atom in structure],
        dtype=bool,
    )
    m = matched & considered
    u = (~matched) & considered

    if int(np.sum(m)) == 0 or int(np.sum(u)) == 0:
        return []

    frac = safe_scaled_positions(structure)
    suggestions = []
    for ax in range(3):
        mv = np.asarray(frac[m, ax], dtype=float)
        uv = np.asarray(frac[u, ax], dtype=float)
        if len(mv) == 0 or len(uv) == 0:
            continue
        matched_min = float(np.min(mv))
        unmatched_max = float(np.max(uv))
        gap = matched_min - unmatched_max
        # Need a meaningful clean lower-bound separation.
        if gap >= 0.01:
            margin = min(0.002, 0.20 * gap)
            threshold = unmatched_max + margin
            suggestions.append(
                RestrictionSuggestion(
                    axis=AXIS_NAME[ax].upper(),
                    threshold=float(threshold),
                    matched_min=matched_min,
                    unmatched_max=unmatched_max,
                    gap=float(gap),
                    confidence="high" if gap >= 0.03 else "moderate",
                )
            )
    return suggestions


# -----------------------------------------------------------------------------
# Recommendation model
# -----------------------------------------------------------------------------

def recommend_geometric_parameters(
    d1: Optional[float],
    d2: Optional[float],
    min_gap: Optional[float],
    future_dmin_median: Optional[float],
) -> Dict[str, Optional[float] | int | str]:
    if d1 is None:
        return {
            "occupancy_tol": 0.60,
            "min_dist": 1.10,
            "connect_cutoff": 2.00,
            "cn_cutoff": 2.00,
            "min_neighbors": 1,
            "interface_mode": "unknown",
            "mobile_radius": 5.0,
        }

    # For a covalent continuation, the first future column is often the most
    # relevant local geometric reference. Use it to protect against global-shell
    # artefacts caused by finite-cell edges.
    d_bond_ref = float(d1)
    if (
        future_dmin_median is not None
        and np.isfinite(future_dmin_median)
        and future_dmin_median <= max(1.45 * d1, d1 + 0.60)
    ):
        d_bond_ref = max(d_bond_ref, float(future_dmin_median))

    occupancy = min(0.75, max(0.45, 0.40 * d_bond_ref))
    occupancy = round_step(occupancy, 0.05)

    min_dist = max(0.70, 0.72 * d_bond_ref)
    min_dist = min(min_dist, 0.82 * d_bond_ref)
    min_dist = round_step(min_dist, 0.05)

    if d2 is not None and d2 > d1 + 0.10:
        shell_cut = d_bond_ref + 0.35 * (d2 - d_bond_ref)
        shell_cut = min(shell_cut, d2 - max(0.10, 0.07 * (d2 - d_bond_ref)))
    else:
        shell_cut = 1.20 * d_bond_ref

    shell_cut = max(shell_cut, 1.12 * d_bond_ref)
    shell_cut = round_step(shell_cut, 0.05, mode="up")

    interface_mode = "covalent-continuation"
    min_neighbors = 1
    connect_cutoff = shell_cut

    if future_dmin_median is not None and np.isfinite(future_dmin_median):
        # If the first future crystallographic layer is far beyond the ordinary
        # first-neighbour bond shell, it is better interpreted as detached-layer
        # nucleation (e.g. graphite stacking) than as a covalent continuation.
        if future_dmin_median > max(1.45 * d1, d1 + 0.60):
            interface_mode = "detached-layer-nucleation"
            min_neighbors = 0
            connect_cutoff = shell_cut
        else:
            # Ensure the actual first future site is not rejected after modest
            # surface relaxation, while still remaining below second shell.
            target = future_dmin_median + max(0.12, 0.08 * d_bond_ref)
            if d2 is not None:
                target = min(target, d2 - 0.12)
            connect_cutoff = max(shell_cut, target)
            connect_cutoff = round_step(connect_cutoff, 0.05, mode="up")

    cn_cutoff = shell_cut

    mobile_radius = max(4.0, 2.5 * d_bond_ref)
    mobile_radius = round_step(mobile_radius, 0.5, mode="up")

    return {
        "occupancy_tol": float(occupancy),
        "min_dist": float(min_dist),
        "connect_cutoff": float(connect_cutoff),
        "cn_cutoff": float(cn_cutoff),
        "min_neighbors": int(min_neighbors),
        "interface_mode": str(interface_mode),
        "mobile_radius": float(mobile_radius),
    }


def analyze_axis(
    structure: Atoms,
    template: Atoms,
    axis: str,
    direction: str,
    species: Optional[set],
) -> AxisReport:
    g = AXIS[axis]

    work_template = filter_atoms_by_species(template, species)
    work_structure = filter_atoms_by_species(structure, species)

    pbc_i, pbc_diagnostics = suggest_pbc_with_diagnostics(
        work_structure, g, species=None
    )
    pbc = tuple(bool(x) for x in pbc_i)

    if len(work_template) == 0:
        raise ValueError(f"No template atoms remain for species filter on axis {axis}.")
    if len(work_structure) == 0:
        raise ValueError(f"No structure atoms remain for species filter on axis {axis}.")

    column_tol, raw_min_gap, _ = estimate_column_tol(work_template, g)
    groups = axis_columns(work_template, g, column_tol)
    counts = [len(grp) for grp in groups]
    coords = [float(np.mean(work_template.positions[grp, g])) for grp in groups]

    if len(groups) >= 2:
        diffs = np.diff(coords)
        min_gap = float(np.min(diffs))
        med_gap = float(np.median(diffs))
    else:
        min_gap = raw_min_gap
        med_gap = None

    gap_pattern, gap_period, gap_source = detect_gap_pattern(coords, column_tol)
    structural_period, structural_source, structural_confidence, _ = best_structural_period(
        work_template, groups, g, signature_tol=max(0.005, 0.05 * column_tol)
    )

    ref_count = representative_column_count(counts)
    edge_count = counts[-1] if direction == "plus" else counts[0]
    edge_completeness = float(edge_count / ref_count) if ref_count > 0 else 0.0
    truncated = bool(ref_count > 0 and edge_completeness < 0.75)

    oblique = cartesian_growth_axis_is_oblique(work_template, g)
    rectangularize = bool(
        axis in {"x", "y"}
        and (truncated or oblique)
        and is_hexagonal_xy(work_template)
    )

    d1, d2, shells = first_two_neighbor_shells(work_structure, pbc, species=None)

    future_pts, future_coord, _ = extrapolate_first_future_column(
        work_template,
        groups,
        g,
        structural_period,
        gap_pattern,
        direction,
    )

    if len(future_pts):
        future_dmins = min_distances_to_structure(work_structure, future_pts, pbc, species=None)
        fdmin = float(np.min(future_dmins))
        fdmed = float(np.median(future_dmins))
        fdmax = float(np.max(future_dmins))
    else:
        future_dmins = np.asarray([], dtype=float)
        fdmin = fdmed = fdmax = None

    rec = recommend_geometric_parameters(d1, d2, min_gap, fdmed)
    connect_cutoff = float(rec["connect_cutoff"]) if rec["connect_cutoff"] is not None else None

    future_counts = []
    if len(future_pts) and connect_cutoff is not None:
        future_counts = neighbor_counts_for_points(
            work_structure,
            future_pts,
            pbc,
            connect_cutoff,
            species=None,
        )

    if min_gap is not None:
        layer_tol = min(0.40, 0.45 * min_gap)
        layer_tol = max(0.05, round_step(layer_tol, 0.05, mode="down"))
    else:
        layer_tol = 0.25

    front_min = 0.05
    # In shell mode front_max is not the principal selector in the current mc_flux
    # implementation, but this remains a useful safe value if global mode is used.
    if gap_pattern:
        front_max = max(3.0, min(8.0, 2.5 * max(gap_pattern)))
        front_max = round_step(front_max, 0.5, mode="up")
    else:
        front_max = 3.5

    warnings: List[str] = []
    notes: List[str] = []

    warnings.extend(pbc_diagnostics)

    if truncated:
        warnings.append(
            f"Terminal {axis.upper()} column contains {edge_count} atoms whereas the "
            f"representative complete column contains {ref_count} "
            f"(edge completeness {edge_completeness:.1%})."
        )
    if oblique and axis in {"x", "y"}:
        warnings.append(
            f"The lattice is oblique with respect to Cartesian {axis.upper()}; "
            "Cartesian edge columns may not represent complete crystallographic rows."
        )
    if rectangularize:
        warnings.append(
            "Hexagonal/oblique lateral growth detected. Convert the basal cell to an "
            "orthogonal rectangular XY supercell before production growth."
        )
    if structural_source == "cell-fallback":
        notes.append(
            "No shorter structural motif was demonstrated internally; the advisor uses "
            f"the complete observed {len(groups)}-column sequence as the periodic closure."
        )
    if rec["interface_mode"] == "detached-layer-nucleation":
        warnings.append(
            "The first future layer is much farther from the current structure than the "
            "ordinary first-neighbour bond length. This is classified as detached-layer "
            "nucleation; --template-min-neighbors 0 is recommended rather than inflating "
            "the covalent connectivity cutoff."
        )
    if d1 is not None and connect_cutoff is not None and connect_cutoff <= d1 * 1.03:
        warnings.append(
            "Recommended connectivity margin is very small. Increase the cutoff slightly "
            "if local relaxation causes physically bonded sites to be rejected."
        )

    if shells:
        notes.append(
            "Detected neighbour shells using validated PBC (first few): "
            + ", ".join(f"{d:.3f} Å" for d, _ in shells[:4])
        )
    if fdmed is not None:
        notes.append(
            f"First future-column interface distance: median={fdmed:.3f} Å "
            f"(min={fdmin:.3f} Å, max={fdmax:.3f} Å)."
        )
    if future_counts:
        notes.append(
            "Ideal first-future-column neighbour counts at recommended cutoff: "
            f"min={min(future_counts)}, median={np.median(future_counts):.1f}, max={max(future_counts)}."
        )

    return AxisReport(
        axis=axis,
        growth_direction=direction,
        pbc=pbc_i,
        n_structure_atoms=len(work_structure),
        n_template_atoms=len(work_template),
        n_columns=len(groups),
        column_counts=counts,
        edge_column_count=int(edge_count),
        reference_column_count=int(ref_count),
        edge_completeness=float(edge_completeness),
        min_column_gap_A=float(min_gap) if min_gap is not None else None,
        median_column_gap_A=float(med_gap) if med_gap is not None else None,
        gap_pattern_A=[float(x) for x in gap_pattern],
        gap_period=int(gap_period) if gap_period else None,
        gap_period_source=gap_source,
        structural_period=int(structural_period) if structural_period else None,
        structural_period_source=structural_source,
        structural_period_confidence=float(structural_confidence),
        first_neighbor_A=float(d1) if d1 is not None else None,
        second_neighbor_A=float(d2) if d2 is not None else None,
        first_future_coord_A=float(future_coord) if future_coord is not None else None,
        first_future_count=int(len(future_pts)),
        first_future_dmin_min_A=float(fdmin) if fdmin is not None else None,
        first_future_dmin_median_A=float(fdmed) if fdmed is not None else None,
        first_future_dmin_max_A=float(fdmax) if fdmax is not None else None,
        future_neighbor_counts=[int(x) for x in future_counts],
        interface_mode=str(rec["interface_mode"]),
        oblique_warning=oblique,
        truncated_edge_warning=truncated,
        rectangularization_recommended=rectangularize,
        recommended_column_tol_A=float(column_tol),
        recommended_layer_tol_A=float(layer_tol),
        recommended_occupancy_tol_A=float(rec["occupancy_tol"]),
        recommended_min_dist_A=float(rec["min_dist"]),
        recommended_connect_cutoff_A=float(rec["connect_cutoff"]),
        recommended_cn_cutoff_A=float(rec["cn_cutoff"]),
        recommended_min_neighbors=int(rec["min_neighbors"]),
        recommended_front_min_A=float(front_min),
        recommended_front_max_A=float(front_max),
        recommended_mobile_radius_A=float(rec["mobile_radius"]),
        warnings=warnings,
        notes=notes,
    )


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

def fmt(value: Optional[float], ndigits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{ndigits}f}"


def print_restrictions(suggestions: Sequence[RestrictionSuggestion]) -> None:
    print("\n" + "=" * 78)
    print("AUTOMATIC FRACTIONAL-RESTRICTION ANALYSIS")
    print("=" * 78)
    if not suggestions:
        print("No clean lower-bound --restrict_X/Y/Z separation was detected.")
        print("This is normal when structure and template contain the same crystal only.")
        return

    print(
        "The structure contains same-species atoms not matched by the supplied template.\n"
        "The following lower-bound fractional cuts cleanly separate the matched template\n"
        "region from those unmatched atoms:"
    )
    for s in suggestions:
        print(
            f"  --restrict_{s.axis} {s.threshold:.6f}    "
            f"(unmatched max={s.unmatched_max:.6f}, matched min={s.matched_min:.6f}, "
            f"gap={s.gap:.6f}, confidence={s.confidence})"
        )


def command_fragment(report: AxisReport, restrictions: Sequence[RestrictionSuggestion]) -> str:
    rmap = {r.axis.lower(): r for r in restrictions}
    lines = [
        f"--growth-axis {report.axis} \\",
        f"--pbc {report.pbc[0]} {report.pbc[1]} {report.pbc[2]} \\",
        "--template-build-mode extend-columns \\",
        f"--template-column-period {report.structural_period} \\",
        f"--template-column-tol {report.recommended_column_tol_A:.2f} \\",
        "--template-front-mode shell \\",
        f"--template-layer-tol {report.recommended_layer_tol_A:.2f} \\",
        f"--template-front-min {report.recommended_front_min_A:.2f} \\",
        f"--template-front-max {report.recommended_front_max_A:.2f} \\",
        f"--template-occupancy-tol {report.recommended_occupancy_tol_A:.2f} \\",
        f"--template-connect-cutoff {report.recommended_connect_cutoff_A:.2f} \\",
        f"--template-min-neighbors {report.recommended_min_neighbors} \\",
        f"--min-dist {report.recommended_min_dist_A:.2f} \\",
        f"--cn-cutoff {report.recommended_cn_cutoff_A:.2f} \\",
        f"--mobile-radius {report.recommended_mobile_radius_A:.1f} \\",
        "--match-cell-to-template \\",
        "--template-debug \\",
    ]

    # Add only restrictions that are useful regardless of the current growth axis.
    for ax in ["x", "y", "z"]:
        if ax in rmap:
            rr = rmap[ax]
            lines.append(f"--restrict_{rr.axis} {rr.threshold:.6f} \\")

    return "\n".join(lines)


def print_axis_report(report: AxisReport, restrictions: Sequence[RestrictionSuggestion]) -> None:
    print("\n" + "=" * 78)
    print(f"GROWTH AXIS: {report.axis.upper()}   direction={report.growth_direction}")
    print("=" * 78)
    print(f"Atoms (structure/template) : {report.n_structure_atoms} / {report.n_template_atoms}")
    print(f"Suggested PBC              : {report.pbc[0]} {report.pbc[1]} {report.pbc[2]}")
    print(f"Detected columns           : {report.n_columns}")
    print(f"Atoms per column           : {report.column_counts}")
    print(
        f"Edge completeness          : {report.edge_column_count}/{report.reference_column_count} "
        f"= {report.edge_completeness:.1%}"
    )
    print(f"Minimum column gap         : {fmt(report.min_column_gap_A)} Å")
    print(f"Median column gap          : {fmt(report.median_column_gap_A)} Å")
    print(
        "Gap pattern                 : ["
        + ", ".join(f"{x:.6f}" for x in report.gap_pattern_A)
        + f"] Å  (period={report.gap_period}, {report.gap_period_source})"
    )
    print(
        f"Structural motif period    : {report.structural_period} "
        f"({report.structural_period_source}, confidence={report.structural_period_confidence:.2f})"
    )
    print(f"1st neighbour shell        : {fmt(report.first_neighbor_A)} Å")
    print(f"2nd neighbour shell        : {fmt(report.second_neighbor_A)} Å")
    print(f"First future coordinate    : {fmt(report.first_future_coord_A)} Å")
    print(f"Atoms in first future col. : {report.first_future_count}")
    print(
        "Future dmin to structure    : "
        f"min={fmt(report.first_future_dmin_min_A)} Å, "
        f"median={fmt(report.first_future_dmin_median_A)} Å, "
        f"max={fmt(report.first_future_dmin_max_A)} Å"
    )
    print(f"Interface classification   : {report.interface_mode}")

    print("\nRecommended geometry-derived starting values")
    print("-" * 78)
    print(f"  --template-column-period      {report.structural_period}")
    print(f"  --template-column-tol         {report.recommended_column_tol_A:.2f}")
    print(f"  --template-layer-tol          {report.recommended_layer_tol_A:.2f}")
    print(f"  --template-front-min          {report.recommended_front_min_A:.2f}")
    print(f"  --template-front-max          {report.recommended_front_max_A:.2f}")
    print(f"  --template-occupancy-tol      {report.recommended_occupancy_tol_A:.2f}")
    print(f"  --template-connect-cutoff     {report.recommended_connect_cutoff_A:.2f}")
    print(f"  --template-min-neighbors      {report.recommended_min_neighbors}")
    print(f"  --min-dist                    {report.recommended_min_dist_A:.2f}")
    print(f"  --cn-cutoff                   {report.recommended_cn_cutoff_A:.2f}")
    print(f"  --mobile-radius               {report.recommended_mobile_radius_A:.1f}")

    if report.warnings:
        print("\nWARNINGS")
        print("-" * 78)
        for w in report.warnings:
            print(f"  ! {w}")

    if report.notes:
        print("\nNotes")
        print("-" * 78)
        for n in report.notes:
            print(f"  - {n}")

    if report.rectangularization_recommended:
        print("\nSuggested preprocessing")
        print("-" * 78)
        print(
            "  The XY cell appears hexagonal/oblique and the Cartesian edge is unsafe.\n"
            "  Convert it to a rectangular supercell before lateral growth, e.g.:\n\n"
            "      python hex_to_rect_xy.py INPUT.cif OUTPUT_rect_xy.cif\n"
        )

    print("\nReady-to-paste mc_flux geometry block")
    print("-" * 78)
    print(command_fragment(report, restrictions))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Analyze a structure/template pair and suggest geometry-derived parameters "
            "for template-constrained growth along x, y and z."
        ),
    )
    p.add_argument("--structure", "--cif", dest="structure", required=True,
                   help="Initial structure (CIF or any ASE-readable format).")
    p.add_argument("--template", "--template-cif", dest="template", default=None,
                   help="Growth template. Defaults to --structure.")
    p.add_argument("--axis", choices=["x", "y", "z", "all"], default="all",
                   help="Analyze one Cartesian growth axis or all three (default: all).")
    p.add_argument("--growth-direction", choices=["plus", "minus"], default="plus",
                   help="Which edge to extrapolate (default: plus).")
    p.add_argument("--species", nargs="+", default=["all"],
                   help="Species participating in growth/template analysis. Default: all.")
    p.add_argument("--match-tol", type=float, default=0.20,
                   help="Å tolerance for structure/template atom matching used to infer restrict_X/Y/Z.")
    p.add_argument("--json", dest="json_path", default=None,
                   help="Optional path for a machine-readable JSON report.")
    p.add_argument("--quiet-command", action="store_true",
                   help="Suppress ready-to-paste command blocks.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    structure_path = Path(args.structure)
    template_path = Path(args.template) if args.template else structure_path
    if not structure_path.exists():
        raise FileNotFoundError(f"Structure not found: {structure_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    structure = read(str(structure_path))
    template = read(str(template_path))
    species = normalize_species_arg(args.species)

    print("=" * 78)
    print("GROWTH PARAMETER ADVISOR")
    print("=" * 78)
    print(f"Structure : {structure_path}")
    print(f"Template  : {template_path}")
    print(f"Species   : {'all' if species is None else sorted(species)}")
    print(f"Direction : {args.growth_direction}")
    print(f"N(structure)={len(structure)}   N(template)={len(template)}")
    print(
        "Cell lengths (structure): "
        + " ".join(f"{x:.6f}" for x in structure.cell.lengths())
        + " Å"
    )
    print(
        "Cell angles  (structure): "
        + " ".join(f"{x:.6f}" for x in structure.cell.angles())
        + " deg"
    )

    restrictions = infer_restrictions(
        structure,
        filter_atoms_by_species(template, species),
        species,
        match_tol_A=float(args.match_tol),
    )
    print_restrictions(restrictions)

    axes = ["x", "y", "z"] if args.axis == "all" else [args.axis]
    reports: List[AxisReport] = []
    for axis in axes:
        report = analyze_axis(
            structure=structure,
            template=template,
            axis=axis,
            direction=args.growth_direction,
            species=species,
        )
        reports.append(report)
        print_axis_report(report, restrictions)

    print("\n" + "=" * 78)
    print("INTERPRETATION / LIMITATIONS")
    print("=" * 78)
    print(
        "These values are geometry-derived starting points. They do not determine DFT/ML\n"
        "accuracy, kinetic barriers, temperature, chemical potentials, adsorption-energy\n"
        "cutoffs, k-point meshes, plane-wave cutoffs, or relaxation convergence. Those\n"
        "quantities require physical validation and convergence tests.\n\n"
        "Recommended production check: run 1--5 growth steps with --template-debug.\n"
        "A healthy setup should have at least one candidate reaching tested_energy, and\n"
        "the first-future-site distance should agree with the expected crystal geometry."
    )

    if args.json_path:
        payload = {
            "structure": str(structure_path),
            "template": str(template_path),
            "species": "all" if species is None else sorted(species),
            "growth_direction": args.growth_direction,
            "structure_cell_lengths_A": [float(x) for x in structure.cell.lengths()],
            "structure_cell_angles_deg": [float(x) for x in structure.cell.angles()],
            "restriction_suggestions": [asdict(x) for x in restrictions],
            "axes": [asdict(x) for x in reports],
        }
        Path(args.json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON report saved to: {args.json_path}")


if __name__ == "__main__":
    main()
