#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Install the repository first: pip install -e .[chgnet] / .[mace] / .[gpaw]
ccgrowth \
  --cif graphene_mos2_seed_x.cif\
  --species Mo S \
  --backend chgnet \
  --build-mode direct \
  --growth-axis x \
  --pbc 0 0 0 \
  --replicate 1 1 1 \
  --vacuum 0 \
  --site-mode template \
  --template-cif mos2_lateral_template_x.cif  \
  --template-species-mode template \
  --template-site-species Mo S \
  --growth-origin-species Mo S \
  --template-build-mode extend-columns \
  --template-extend-ncols 2 \
  --template-column-period 6 \
  --template-column-tol 0.15 \
  --template-growth-direction plus \
  --template-front-mode shell \
  --template-layer-tol 0.25 \
  --template-front-min 0.05 \
  --template-front-max 3.5 \
  --template-occupancy-tol 0.75 \
  --template-connect-cutoff 2.9 \
  --template-min-neighbors 0 \
  --min-dist 1.7 \
  --cn-cutoff 2.9 \
  --match-cell-to-template \
  --cell-growth-margin 5 \
  --skip-bulk-reference \
  --relax-steps 6 \
  --steps 500 \
  --mobile-radius 6.0 \
  --front-grid 3 6 \
  --vacancy-mode frozen \
  --max-adsorption-energy-eV 50.0 \
  --out grafeno_cutoff.traj \
  --log grafeno_cutoff.csv \
  --bulk-energy-repeat 1 1 1 \
  > grafeno_cutoff.log 2>&1
