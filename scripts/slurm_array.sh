#!/bin/bash
# Submit one array job per config:
#   sbatch --array=0-149 scripts/slurm_array.sh configs/superbit.yaml 200
# runs 150 chunks of 200 galaxies = 30,000 galaxies (120,000 images).
#SBATCH --job-name=squarem-shear
#SBATCH --cpus-per-task=1
#SBATCH --mem=3G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail
config=$1
nobj=$2
name=$(basename "$config" .yaml)
chunk=${SLURM_ARRAY_TASK_ID}

# one thread per job; numba and numpy must not oversubscribe the node
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1

out=results/${name}/chunk$(printf %05d "$chunk").npz
if [ -f "$out" ]; then
    echo "$out exists, skipping"
    exit 0
fi
python -m squarem_shear.run "$config" --chunk "$chunk" --nobj "$nobj" --out "$out"
