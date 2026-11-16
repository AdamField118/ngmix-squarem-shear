#!/bin/bash
# Submit one configuration as a SLURM array of NTASKS tasks.
#
#   scripts/submit.sh CONFIG NTASKS HOURS [extra sbatch args...]
#   scripts/submit.sh configs/superbit.yaml 10 12 --partition=long
#
# 150 chunks of 200 galaxies (30,000 galaxies) are spread over NTASKS tasks.
# Choose NTASKS so the total across your arrays stays under the cluster's
# per-user job limit, and HOURS so one task's chunks fit (see run_all.sh for
# measured seconds per galaxy).  Find your limits with
#   sacctmgr -n show assoc user=$USER format=maxsubmit,maxjobs
#   sacctmgr -n show qos format=name%20,maxsubmitpu,maxjobspu,maxwall
set -euo pipefail
config=$1
ntasks=$2
hours=$3
shift 3
NCHUNKS=${NCHUNKS:-150}
NOBJ=${NOBJ:-200}
mkdir -p logs
per_task=$(( (NCHUNKS + ntasks - 1) / ntasks ))
echo "$config: $ntasks tasks x $per_task chunks x $NOBJ galaxies, ${hours}h each"
sbatch --array=0-$((ntasks - 1)) --time="${hours}:00:00" "$@" \
    scripts/slurm_array.sh "$config" "$NOBJ" "$NCHUNKS" "$ntasks"
