# ngmix-squarem-shear

Shear-calibration simulations for *Accelerating Weak-Lensing Shape
Measurement with SQUAREM-Optimized Expectation–Maximization* (Field,
Sanguinet & Saha). The repo measures how the convergence of the EM PSF fit
propagates to metacalibration's multiplicative bias *m*, additive bias *c*,
response *R* and PSF leakage *α*. It uses COSMOS galaxies and PSFs from
several telescopes.

The figures in the paper are made by
[ngmix-squarem-paper-plots](https://github.com/AdamField118/ngmix-squarem-paper-plots)
from the `summary.json` files this repo writes.

## What is simulated

Every galaxy is drawn four times: sheared by g₁ = ±0.02, at rotations of 0°
and 90°. All four images share one noise field. Each image is measured
once per **arm**. The arms are identical except for the PSF fitter:

| arm | PSF fit |
|---|---|
| `em5_default` | 5-gaussian EM, ngmix defaults (`tol=1e-5`, `maxiter=500`) |
| `em5_superbit` | 5-gaussian EM, SuperBIT production settings (`tol=1e-6`, `maxiter=50000`) |
| `em5_squarem` | the same, with `accelerate=True` |
| `em5_converged` | SQUAREM at `tol=1e-10`, the reference |

The measurement copies the SuperBIT pipeline:
- metacal with `psf='dilate'` and all nine metacal types;
- a `gauss` galaxy model fit with LM;
- `ntry=20`.

Every arm reuses the same random seed, so all arms get the same metacal
noise and the same starting guesses. Differences between arms are therefore
paired. They are measured much more precisely than any single arm's
absolute bias.

PSF sources are in `squarem_shear/psfs.py`. The paper uses three:

| source | config | data |
|---|---|---|
| SuperBIT PSFEx models (50 exposures × 20 positions each) | `superbit.yaml` | `.psf` files in `data/psfs/superbit/` |
| HST/ACS F814W PSFs shipped with the COSMOS sample (1000) | `hst.yaml` | COSMOS download |
| Roman H158 design PSFs at random SCA positions (1000) | `roman.yaml` | none (built into GalSim) |

`superbit_ngauss.yaml` is for the appendix. It compares 3, 5, 7 and 9
gaussians on the SuperBIT PSFs.

SuperBIT positions are drawn uniformly over the range each PSFEx model was
fit on, POLZERO ± POLSCAL/2 from its header. The pool cycles through the
files, so every model is used equally often. PSFs are built lazily, so a
job only builds the PSFs its galaxies use.

## Install

```bash
conda env create -f environment.yml
conda activate squarem-shear
```

ngmix is installed from the fork's `em-squarem` branch, pinned to the
audited commit. Once the change is in an ngmix release, replace that line
with `ngmix>=X.Y`.

## Data

```bash
scripts/download_data.sh            # COSMOS 25.2 training sample (~4 GB)
mkdir -p data/psfs/superbit && cp /path/to/superbit/*.psf data/psfs/superbit/
```

## Run

Run the smoke test first. It needs no data and takes a few minutes:

```bash
pytest tests
python -m squarem_shear.run configs/smoke.yaml --chunk 0 --nobj 10 --out results/smoke/chunk00000.npz
python -m squarem_shear.collect results/smoke/chunk*.npz --out results/smoke/summary.json --reference em5_converged
```

Then the full runs. Each line of `scripts/run_all.sh` is one SLURM array
of 150 × 200 = **30,000 galaxies (120,000 images) per PSF source**:

```bash
scripts/run_all.sh
scripts/collect_all.sh
```

A chunk is skipped if its output already exists, so failed array tasks can
simply be resubmitted. Chunk *c*, object *i* uses seed
`seed + 1_000_000*c + i`, so any single object can be regenerated on its own.

**Before the production runs, tune each config's `noise_sigma`, `flux_scale`
and `selection.min_true_flux`.** `collect` prints the S/N quantiles of the
kept sample. Aim for the S/N distribution of the survey you want to
represent. Galaxies at S/N ≲ 7 make the per-object responses wildly noisy.
The cut is on the *true* flux, so it cannot introduce shear-dependent
selection.

Cost: about 10–30 s per galaxy per core for the four arms, mostly in the
plain-EM arms.

## Estimators

These are implemented in `squarem_shear/collect.py`. For each galaxy:
- D is half the ±g difference in g₁, averaged over the two rotations;
- S is the mean of (g₁, g₂) over the four images.

From these:

    m = ⟨D⟩ / (g ⟨R₁₁⟩) − 1
    cᵢ = ⟨Sᵢ⟩ / ⟨Rᵢᵢ⟩
    α = slope of S/⟨R⟩ against the native PSF's adaptive-moment ellipticity

Errors come from bootstrapping galaxies. The same resamples are used for
every arm, so the `d*` fields (arm − reference) have paired errors.

`summary.json` also records the cost of each arm, per object (four metacal
measurements of nine PSF fits each). Each quantity is given as a mean,
median, standard deviation and 5th/95th percentiles over all 30,000 objects:
- wall time;
- the fraction of it spent in EM PSF fits;
- EM iterations per PSF fit.

It also records the fraction of PSF fits that hit `maxiter`. These
replace the five-object timing quoted in early drafts.

## Layout

```
squarem_shear/   psfs.py galaxies.py sim.py measure.py guessers.py run.py collect.py
configs/         superbit.yaml hst.yaml roman.yaml superbit_ngauss.yaml smoke.yaml
scripts/         SLURM array, run_all / collect_all, data download
tests/           smoke test
```

`guessers.py` provides starting mixtures for more than five gaussians.
ngmix's `GMixPSFGuesser` stops at five.
