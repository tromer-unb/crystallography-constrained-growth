# HPC execution notes

The production calculations were launched with PBS scripts on a local cluster. Those scripts mixed scientific parameters with machine-specific concerns: queue names, absolute Conda paths, thread environment variables, and, in one MACE script, package installation during the job.

For reproducibility, this repository treats `cases/*/run.sh` as the scientific source of truth. A scheduler wrapper should activate a pre-built environment and call the case script. Do not install or upgrade packages inside a production job; build and record the environment first.

A minimal PBS wrapper can therefore look like:

```bash
#!/bin/bash
#PBS -N ccgrowth
#PBS -q workq
#PBS -l nodes=1:ppn=1
set -euo pipefail
source /path/to/conda.sh
conda activate ccgrowth-chgnet
cd "$PBS_O_WORKDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash cases/graphene/run.sh
```

Adapt queue/resource directives and MPI launch strategy to the backend and cluster installation.
