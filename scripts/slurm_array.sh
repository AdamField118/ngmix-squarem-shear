#!/bin/bash
# One SLURM array task: runs every chunk c in 0..nchunks-1 with
# c % ntasks == task id, one after another.  Packing chunks into fewer
# tasks keeps an array under per-user job limits; the outputs are the same
# as one chunk per task.  Chunks already written are skipped, so a failed or
# timed-out task can simply be resubmitted.
#
# Submit with scripts/submit.sh, which sets --array and --time.
#SBATCH --job-name=squarem-shear
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
config=$1
nobj=$2
nchunks=$3
ntasks=$4
name=$(basename "$config" .yaml)
task=${SLURM_ARRAY_TASK_ID}

# one thread per job; numba and numpy must not oversubscribe the node
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1

for ((chunk = task; chunk < nchunks; chunk += ntasks)); do
    out=results/${name}/chunk$(printf %05d "$chunk").npz
    if [ -f "$out" ]; then
        echo "$out exists, skipping"
        continue
    fi
    python -m squarem_shear.run "$config" --chunk "$chunk" --nobj "$nobj" \
        --out "$out"
done
