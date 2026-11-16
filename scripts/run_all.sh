#!/bin/bash
# The shear-calibration runs in the paper: 30,000 galaxies (120,000 images)
# per PSF source.  Measured cost per galaxy (all four arms, one core):
#   superbit ~8 s, roman ~8.5 s, hst ~62 s, superbit_ngauss ~16 s
# so with the task counts below each task needs about
#   superbit 15 chunks ~ 7 h, roman 15 chunks ~ 7 h,
#   superbit_ngauss 10 chunks ~ 9 h, hst 3 chunks ~ 10.5 h
# and the four arrays total 85 jobs.  Adjust to your job limit and partition
# time limit (see submit.sh); pass sbatch options such as --partition after
# the hours, or set them here with EXTRA.
set -euo pipefail
EXTRA=${EXTRA:-}
scripts/submit.sh configs/superbit.yaml 10 12 $EXTRA
scripts/submit.sh configs/roman.yaml 10 12 $EXTRA
scripts/submit.sh configs/superbit_ngauss.yaml 15 14 $EXTRA
scripts/submit.sh configs/hst.yaml 50 16 $EXTRA
