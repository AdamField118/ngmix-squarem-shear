#!/bin/bash
# COSMOS galaxies (GalSim training sample, ~4 GB for 25.2) into the GalSim
# share directory, or into DIR if given.
set -euo pipefail
if [ $# -ge 1 ]; then
    galsim_download_cosmos -s 25.2 -d "$1"
else
    galsim_download_cosmos -s 25.2
fi
