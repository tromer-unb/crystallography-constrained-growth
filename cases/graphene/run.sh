#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Install the repository first: pip install -e .[chgnet] / .[mace] / .[gpaw]
ccgrowth \
  --cif grafeno_retangular.cif \
  --species C \
  --backend chgnet \
  --build-mode direct \
  --growth-axis x \
  --pbc 0 1 0 \
  --replicate 1 1 1 \
  --vacuum 0 \
  --site-mode template \
  --template-cif grafeno_retangular.cif \
  --template-species-mode template \
  --template-site-species C \
  --growth-origin-species C \
  --template-build-mode extend-columns \
  --template-extend-ncols 50 \
  --template-column-period 2 \
  --template-column-tol 0.2 \
  --template-debug \
  --template-growth-direction plus \
  --template-front-mode shell \
  --template-layer-tol 0.4 \
  --template-front-min 0.05 \
  --template-front-max 3.5 \
  --template-occupancy-tol 0.55 \
  --template-connect-cutoff 1.8 \
  --template-min-neighbors 1 \
  --min-dist 1.0 \
  --cn-cutoff 1.8 \
  --no-relax-first-added \
  --match-cell-to-template \
  --cell-growth-margin 5 \
  --skip-bulk-reference \
  --relax-steps 6 \
  --steps 500 \
  --mobile-radius 4.0 \
  --front-grid 8 8 \
  --vacancy-mode frozen \
  --max-adsorption-energy-eV 50.0 \
  --out grafeno_cutoff.traj \
  --log grafeno_cutoff.csv \
  --bulk-energy-repeat 1 1 1 \
  > grafeno_cutoff.log 2>&1
