#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Install the repository first: pip install -e .[chgnet] / .[mace] / .[gpaw]
ccgrowth \
  --cif Si_111.cif \
  --species Si \
  --backend gpaw \
  --gpaw-mode pw \
  --gpaw-xc PBE \
  --gpaw-kpts 1 1 1 \
  --gpaw-h 0.20 \
  --gpaw-ecut 250 \
  --gpaw-symmetry off \
  --build-mode direct \
  --growth-axis z \
  --pbc 1 1 0 \
  --replicate 1 1 1 \
  --vacuum 0 \
  --site-mode template \
  --template-cif Si_111.cif \
  --template-species-mode template \
  --template-site-species Si \
  --growth-origin-species Si \
  --template-build-mode extend-columns \
  --template-extend-ncols 4 \
  --template-column-period 6 \
  --template-column-tol 0.17 \
  --template-debug \
  --template-growth-direction plus \
  --template-front-mode shell \
  --template-layer-tol 0.35 \
  --template-front-min 0.05 \
  --template-front-max 6.0 \
  --template-occupancy-tol 0.75 \
  --template-connect-cutoff 2.9 \
  --template-min-neighbors 1 \
  --min-dist 1.7 \
  --cn-cutoff 2.9 \
  --no-relax-first-added \
  --match-cell-to-template \
  --cell-growth-margin 5 \
  --skip-bulk-reference \
  --relax-steps 6 \
  --steps 500 \
  --mobile-radius 6.0 \
  --front-grid 6 6 \
  --vacancy-mode frozen \
  --max-adsorption-energy-eV 50.0 \
  --out grafeno_cutoff.traj \
  --log grafeno_cutoff.csv \
  --bulk-energy-repeat 1 1 1 \
  > grafeno_cutoff.log 2>&1
