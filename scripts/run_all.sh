#!/bin/bash
# The shear-calibration runs in the paper: 30,000 galaxies (120,000 images)
# per PSF source, as 150 chunks of 200.  Each line is one SLURM array.
set -euo pipefail
mkdir -p logs
sbatch --array=0-149 scripts/slurm_array.sh configs/superbit.yaml 200
sbatch --array=0-149 scripts/slurm_array.sh configs/hst.yaml 200
sbatch --array=0-149 scripts/slurm_array.sh configs/roman.yaml 200
# appendix: 3/5/7/9 gaussians on SuperBIT PSFs
sbatch --array=0-149 scripts/slurm_array.sh configs/superbit_ngauss.yaml 200
