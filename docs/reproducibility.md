# Reproducibility guide

## Levels of validation

The repository distinguishes three validation levels. **Archive validation** checks hashes, required files, and internal provenance. **Figure regression** rebuilds Figures 1–7 from archived publication outputs and compares the generated summary tables with reference summaries. **Full simulation reproduction** reruns an atomistic trajectory and therefore requires the relevant energetic backend, compute resources, and the publication command under `cases/`.

A normal CI job intentionally performs the first two levels. It does not attempt hundreds of GPAW, CHGNet, or MACE growth events on hosted CI infrastructure.

## Lightweight validation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install "pip==26.2.1"
pip install -r requirements-validation.txt
pip install -e . --no-deps
python scripts/build_manifest.py --check
python scripts/validate_archive.py
python scripts/reproduce_figures.py --check-reference
pytest -q
```

The hosted reference environment is Python 3.11.16. `requirements-validation.txt` pins the complete lightweight package set that passed the archive, unit/provenance, and Figure 1-7 regression checks on 2026-09-17. Production backend environments are deliberately not collapsed into this lock file because GPAW, CHGNet, and MACE have distinct hardware and installation constraints.

`ccgrowth-advisor` is also smoke-tested on the Si(111) publication structure; the expected structural motif is six columns.

## Full growth trajectories

Install the backend required by the case:

```bash
pip install -e ".[chgnet]"  # CHGNet cases
pip install -e ".[mace]"    # MACE cases
pip install -e ".[gpaw]"    # Python package; GPAW also needs a valid local installation/setup
```

Then execute a case from the repository root, e.g. `bash cases/graphene/run.sh`. The scripts preserve the scientific CLI arguments from the production PBS jobs but omit machine-specific queue directives, absolute Conda paths, and in-job package installation.

## Randomness and ensembles

Production commands record explicit seeds when they were supplied. A common seed is useful for backend-portability comparisons, but identical seeds do not force identical trajectories once energy models assign different candidate probabilities. Defect trajectories should be replicated over independent seeds when estimating uncertainty.

## Figure regression

`python scripts/reproduce_figures.py --check-reference` sets `MPLBACKEND=Agg`, runs each publication figure script, and compares its generated summary CSV with `results/figure_summaries/`. Numeric values must agree to a strict tolerance. This makes the figure pipeline testable without visual inspection alone.
