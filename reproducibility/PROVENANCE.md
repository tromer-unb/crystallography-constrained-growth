# Provenance and restructuring record

Source material was audited from `/home/tromer/growth_crystal/PAPER` on 2026-09-17 before restructuring.

The working tree contained 458 files and 47 Python files. Content hashing showed that 26 copies of the main growth engine were byte-for-byte identical. The repository therefore retains a single canonical copy at `src/ccgrowth/growth.py`. Two generations of the growth-parameter advisor were present; the newer implementation, including periodic-seam validation and updated first-neighbor diagnostics, is retained as `src/ccgrowth/parameter_advisor.py`.

The historical `figure1/figure1.py` generated the manuscript's Si-growth figure (Figure 2), while `figure2/figure2.py` generated the manuscript's automatic-parameterization figure (Figure 1). They are renamed in this repository to match the manuscript numbering.

Before restructuring, all seven original figure scripts executed successfully with `MPLBACKEND=Agg`. After relocation and path normalization, all seven scripts executed again. Their generated summary tables matched the original summary tables column-by-column, with numerical comparisons performed at `rtol=atol=1e-12` after excluding path-only metadata.

The Si(110) GPAW archive has one known state-file inconsistency: the final event table reports 109 atoms; `grafeno_cutoff_checkpoint.xyz` has 109 atoms; the historical `grafeno_cutoff_final.xyz` has 82 atoms. The checkpoint is therefore the authoritative structure for the publication pipeline. The 82-atom file is retained with the explicit suffix `_legacy_mismatch` for auditability.
