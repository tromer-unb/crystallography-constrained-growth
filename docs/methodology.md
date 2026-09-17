# Methodology map

The implementation is event based. At accepted step `n`, the current atomic configuration `C_n` is converted into a finite set of admissible candidates `A_n`. In template-constrained calculations, `A_n` is inherited from a crystallographic continuation manifold. In template-free calculations, it is generated from distributed local surface sampling. The energetic backend evaluates the surviving candidates, a stochastic rule selects an event, and a local relaxation produces `C_{n+1}`.

The important software separation is therefore:

`candidate generation -> geometric screening -> energetic evaluation -> stochastic selection -> local relaxation -> logging`

`src/ccgrowth/growth.py` implements this common engine. The publication uses GPAW, CHGNet, and MACE through calculator interfaces. Additional calculator hooks exist in the code, but they are not required to reproduce the reported manuscript cases.

## Crystallographic parameter advisor

`src/ccgrowth/parameter_advisor.py` analyzes the structure/template before production. It determines the growth-axis column sequence, the periodic gap pattern, a structural motif period, PBC suggestions, and geometric cutoffs. The canonical version retained in this repository includes seam-distance validation: a transverse direction is not declared periodic simply because atoms span the fractional cell if wrapping that seam would create an unphysical first-neighbor distance.

For the Si(111) publication input, the advisor identifies an alternating gap pattern near `2.357/0.786 Å` and a six-column structural motif. The distinction between the gap period and the structural motif period is essential to avoid a geometrically invalid continuation.

## Structural observables

The archived CSV event tables record the quantities used in the paper, including accepted species/event identity, unrelaxed insertion energy, tested-candidate counts and selection probabilities, growth-front statistics, template occupation, coordination metrics, defect counts, and backend-specific trajectory metadata where applicable. Figure scripts consume these records and the archived atomic structures directly; they do not reconstruct idealized lattices for the quantitative panels.

## Scope

The trajectories are structurally and energetically informed growth pathways. Without explicit transition-state barriers and calibrated attempt frequencies, their internal stochastic clock should not be interpreted as a direct experimental timescale. Defect-containing trajectories are particularly path dependent; quantitative defect statistics require ensembles over independent random seeds.
