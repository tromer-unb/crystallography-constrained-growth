# Data layout and archival policy

`results/` is a curated, read-only snapshot of the publication runs. It contains the event-level CSV tables, textual logs used to recover run metadata, initial/final/checkpoint XYZ structures, recorded vacancy coordinates, and the one initial CIF required directly by the Figure 1 active-mask analysis.

`cases/` contains inputs and commands needed to launch new runs. New simulations should write to a separate work directory rather than overwrite `results/`.

`reproducibility/manifest.csv` is the integrity boundary for both directories. The validator requires exact manifest coverage: an archived file that is missing, modified, duplicated in the manifest, or newly added without a checksum causes validation to fail.

## Large trajectories

ASE `.traj` files from the working directory are substantially larger than the figure-level archive and contain intermediate atomistic frames. They are deliberately not committed to ordinary Git history. For a journal data-availability package, deposit those raw trajectories in a DOI-backed research-data archive and publish a checksum manifest.

A recommended external archive layout is:

```text
raw-trajectories/<case-name>/*.traj
checksums.sha256
environments/<backend>.txt
README.md
```

The archive README should map every trajectory to the corresponding `cases/<case-name>/run.sh`, state the software/backend version, random seed where applicable, and hardware/runtime information needed to interpret the calculation. Record the persistent DOI here and in `CITATION.cff` only after the archive is actually minted; do not use a provisional or invented identifier.

The Git repository already preserves the event histories and atomic states used by Figures 1-7, so the external trajectory deposit is an additional raw-data layer rather than a prerequisite for checking the published figure pipeline.

## Provenance exception: Si(110) GPAW

The original working tree contained a `grafeno_cutoff_final.xyz` with 82 atoms while the final event-table row reports 109 atoms. The 109-atom checkpoint agrees with the event table and is the structure used by the figure code. The inconsistent 82-atom artifact is retained as `grafeno_cutoff_final_legacy_mismatch.xyz` rather than silently overwritten; see `reproducibility/PROVENANCE.md`.
