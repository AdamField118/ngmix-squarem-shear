#!/bin/bash
# Summaries for the paper.  Copy results/<source>/summary.json to
# ngmix-squarem-paper-plots/data/shear/<source>/summary.json.
set -euo pipefail
for d in results/*/; do
    python -m squarem_shear.collect "$d"/chunk*.npz \
        --out "$d"/summary.json --reference em5_converged
done
