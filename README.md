# Crystallography-Constrained Growth

Reproducibility repository for **A General Atomistic Framework for Crystallography-Constrained Crystal Growth**. The code separates the construction of a physically admissible candidate manifold from energetic ranking, stochastic selection, and local relaxation. The same event engine is used for crystallographic growth and template-free deposition; the energetic backend can be GPAW, CHGNet, or MACE for the simulations reported in the manuscript.

## Reproducibility contract

This repository is organized around four layers that are intentionally kept separate:

1. **`src/ccgrowth/`** — the canonical growth engine and automatic growth-parameter advisor. There is one source of truth; case directories do not carry private copies of the engine.
2. **`cases/`** — publication cases, input structures, and portable commands corresponding to the production runs.
3. **`results/`** — immutable, curated publication outputs used to rebuild the figures: event tables, logs, initial/final/checkpoint structures, and defect records.
4. **`figures/`** — scripts for manuscript Figures 1–7, numbered exactly as in the paper.

The historical working directory contained many identical copies of the growth script. Those copies are intentionally consolidated here. The archived numerical outputs are not recomputed or altered by that refactor.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "pip==26.2.1"
pip install -r requirements-validation.txt
pip install -e . --no-deps

ccgrowth --help
ccgrowth-advisor --help
python scripts/build_manifest.py --check
python scripts/validate_archive.py
python scripts/reproduce_figures.py --check-reference
pytest -q
```

The manifest check first verifies that the committed archive index is deterministic. The figure pipeline executes all seven plotting programs with a non-interactive Matplotlib backend and verifies their summary tables against the publication reference tables. `requirements-validation.txt` is the locked lightweight environment used by hosted CI (Python 3.11.16); it is intentionally separate from the much heavier GPAW/CHGNet/MACE production environments.

## Running a growth case

Each directory below `cases/` contains the crystallographic input(s) and the exact publication command stripped of cluster-specific `PBS` and Conda setup. Install the backend needed by that case and run its script, for example:

```bash
pip install -e ".[chgnet]"
bash cases/graphene/run.sh
```

For GPAW and MACE production calculations, use an environment appropriate for your machine/HPC installation. The original PBS environment was machine-specific and is documented rather than treated as a portable dependency specification.

## Paper-to-code map

| Manuscript item | Reproduction script | Main archived systems |
|---|---|---|
| Fig. 1 — automatic crystallographic parameterization | `figures/fig01_parameter_advisor.py` | graphene X/Y, Si(111), graphite, graphene/SiC |
| Fig. 2 — facet-resolved first-principles Si growth | `figures/fig02_si_gpaw_facets.py` | Si(100), Si(110), Si(111), GPAW |
| Fig. 3 — carbon structural generality | `figures/fig03_carbon_architectures.py` | graphene, diamond, graphite, CHGNet |
| Fig. 4 — heterogeneous/multicomponent growth | `figures/fig04_heterointerfaces.py` | graphene/SiC, MoS2/graphene |
| Fig. 5 — programmable point disorder | `figures/fig05_defects.py` | pristine, vacancy, vacancy+N graphene |
| Fig. 6 — backend portability | `figures/fig06_backend_portability.py` | Si facets with GPAW/CHGNet/MACE |
| Fig. 7 — template-free deposition | `figures/fig07_template_free.py` | Fe/B on finite bilayer graphene |

The original working tree had the first two local figure directories reversed relative to the manuscript. The repository names above correct that provenance ambiguity.

## Citation

`CITATION.cff` provides the software citation metadata and redirects the preferred scholarly citation to the manuscript **A General Atomistic Framework for Crystallography-Constrained Crystal Growth**. The current manuscript PDF identifies the authoring entity as `LCCMat UnB` (2026). When the final article author list, DOI, journal metadata, or a versioned data DOI becomes available, update those fields explicitly rather than adding provisional identifiers.

## What is and is not archived in Git

The repository contains the information required to audit the reported event histories and regenerate all manuscript figures. Large ASE `.traj` files, scheduler stdout/stderr, and generated raster/vector figure files are not duplicated in ordinary Git history. They are derived or bulky artifacts; for long-term raw-trajectory preservation, use a versioned data release or research-data archive and record its DOI in `docs/data-layout.md`.

See `docs/reproducibility.md`, `docs/methodology.md`, and `reproducibility/PROVENANCE.md` for the detailed audit trail.
