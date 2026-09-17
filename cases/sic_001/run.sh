#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Install the repository first: pip install -e .[chgnet] / .[mace] / .[gpaw]
ccgrowth \
  --cif SiC_grafeno.cif \
  --species C \
  --backend chgnet \
  --build-mode direct \
  --growth-axis x \
  --pbc 0 0 0 \
  --replicate 1 1 1 \
  --vacuum 0 \
  --site-mode template \
  --template-cif hex_grafeno.cif \
  --template-species-mode template \
  --template-site-species C \
  --growth-origin-species C \
  --template-build-mode extend-columns \
  --template-extend-ncols 20 \
  --template-column-period 2 \
  --template-column-tol 0.2 \
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
  --match-cell-to-template \
  --cell-growth-margin 5 \
  --skip-bulk-reference \
  --relax-steps 6 \
  --steps 500 \
  --restrict_Z 0.37418 \
  --mobile-radius 4.0 \
  --front-grid 3 6 \
  --vacancy-mode frozen \
  --max-adsorption-energy-eV 50.0 \
  --out grafeno_cutoff.traj \
  --log grafeno_cutoff.csv \
  --bulk-energy-repeat 1 1 1 \
  > grafeno_cutoff.log 2>&1
