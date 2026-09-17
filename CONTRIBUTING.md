# Contributing

Keep scientific logic, publication inputs, immutable results, and plotting code separate. Changes to `src/ccgrowth/` should include a test or an explicit reproducibility rationale. Do not overwrite files in `results/`; add a new named run or release instead. New publication-facing figures should derive quantitative values from archived tables/structures rather than hard-coded values.

`reproducibility/manifest.csv` is generated deterministically from `cases/` and `results/`. If, and only if, a deliberate archive change is made, regenerate it with `python scripts/build_manifest.py` and review the checksum diff. Ordinary code or documentation edits must not change the archive manifest.

For a proposed scientific change, open a branch, document the affected cases, run `make check`, then open a pull request describing whether numerical outputs are expected to change. Use `make install-validation` when you need the exact lightweight environment used by hosted reproducibility CI; backend-specific production environments should be recorded separately.
