#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Install the repository first: pip install -e .[chgnet] / .[mace] / .[gpaw]
ccgrowth \
  --cif graphit_replicado_vacuo.cif \
  --species Fe B \
  --flux-species Fe 1.0 B 1.0 \
  --mu-species Fe -0.4 B -0.1 \
  --backend chgnet \
  --build-mode direct \
  --replicate 1 1 1 \
  --growth-axis z \
  --pbc 0 0 0 \
  --site-mode random \
  --surface-grid 8 8 \
  --deposition-candidates 1 \
  --anchor-species C \
  --anchor-fallback stop \
  --droplet-prob 0.85 \
  --droplet-candidates 48 \
  --droplet-anchor-mode added \
  --side-radius-min 1.7 \
  --side-radius-max 2.6 \
  --side-height-jitter 0.20 \
  --coord-bonus 0.15 \
  --vertical-penalty 0.60 \
  --temperature 300 \
  --height-min 1.2 \
  --height-max 1.7 \
  --min-dist 1.35 \
  --local-top-radius 2.5 \
  --surface-window 3.0 \
  --cn-cutoff 2.8 \
  --mobile-radius 4.5 \
  --relax-steps 40 \
  --fmax 0.05 \
  --skip-bulk-reference \
  --max-adsorption-energy-eV 50.0 \
  --steps 500 \
  --front-grid 8 8 \
  --seed 7 \
  --out graphite_FeB_lateral.traj \
  --log graphite_FeB_lateral.csv \
  > graphite_FeB_lateral.log 2>&1
