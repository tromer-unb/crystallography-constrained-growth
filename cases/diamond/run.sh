#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Install the repository first: pip install -e .[chgnet] / .[mace] / .[gpaw]
ccgrowth \
  --cif diamond_332.cif \
  --species C \
  --backend chgnet \
  --build-mode direct \
  --growth-axis z \
  --pbc 1 1 0 \
  --replicate 1 1 1 \
  --vacuum 0 \
  --site-mode template \
  --template-cif diamond_332.cif \
  --template-species-mode template \
  --template-site-species C \
  --growth-origin-species C \
  --template-build-mode extend-columns \
  --template-extend-ncols 50 \
  --template-column-period 4 \
  --template-column-tol 0.2 \
  --template-debug \
  --template-growth-direction plus \
  --template-front-mode shell \
  --template-layer-tol 0.4 \
  --template-front-min 0.05 \
  --template-front-max 3.0 \
  --template-occupancy-tol 0.6 \
  --template-connect-cutoff 1.9 \
  --template-min-neighbors 1 \
  --min-dist 1.1 \
  --cn-cutoff 1.9 \
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
  --out diamond_cut.traj \
  --log diamond_cut.csv \
  --bulk-energy-repeat 1 1 1 \
  > diamond.log 2>&1
