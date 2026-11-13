"""
Run one chunk of a simulation.

    python -m squarem_shear.run configs/roman.yaml --chunk 0 --nobj 100 \
        --out results/roman/chunk0000.npz

Chunk c, object i uses seed base_seed + 1_000_000*c + i for the galaxy,
images and every arm's metacal rng, so any object can be regenerated alone
and the arms are paired.
"""
import argparse
import os
import time
import numpy as np
import yaml

from .psfs import build_pool
from .galaxies import GalaxySource
from .sim import make_images, psf_moments, ROT_ANGLES, SHEAR_SIGNS
from .measure import measure

NIMAGE = len(ROT_ANGLES) * len(SHEAR_SIGNS)
PER_IMAGE = ('flags', 'g', 'R', 'Rpsf', 's2n', 'T')
PER_ARM = ('psf_time', 'psf_nfit', 'psf_numiter', 'psf_nmaxiter',
           'total_time')


def run_chunk(conf, chunk, nobj, verbose=True):
    pool = build_pool(conf)
    pmom = {}
    galsrc = GalaxySource(conf)
    arms = conf['arms']
    base_seed = conf.get('seed', 1234)

    out = {
        'seed': np.zeros(nobj, dtype='i8'),
        'psf_index': np.zeros(nobj, dtype='i4'),
        'psf_e': np.zeros((nobj, 2)),
        'psf_T': np.zeros(nobj),
        'gal_index': np.zeros(nobj, dtype='i8'),
        'gal_flux': np.zeros(nobj),
        'gal_hlr': np.zeros(nobj),
        'sign': np.array([im_sign for _, im_sign in _layout()]),
        'rot': np.array([im_rot for im_rot, _ in _layout()]),
    }
    for arm in arms:
        n = arm['name']
        out[n + '/flags'] = np.zeros((nobj, NIMAGE), dtype='i4')
        out[n + '/g'] = np.full((nobj, NIMAGE, 2), np.nan)
        out[n + '/R'] = np.full((nobj, NIMAGE, 2, 2), np.nan)
        out[n + '/Rpsf'] = np.full((nobj, NIMAGE, 2, 2), np.nan)
        out[n + '/s2n'] = np.full((nobj, NIMAGE), np.nan)
        out[n + '/T'] = np.full((nobj, NIMAGE), np.nan)
        for k in PER_ARM:
            out[n + '/' + k] = np.zeros((nobj, NIMAGE))

    t0 = time.time()
    for i in range(nobj):
        seed = base_seed + 1_000_000 * chunk + i
        rng = np.random.RandomState(seed)
        ipsf = rng.randint(len(pool))
        gal, grec = galsrc.draw(rng)
        images = make_images(conf, gal, pool[ipsf], rng)

        out['seed'][i] = seed
        out['psf_index'][i] = ipsf
        if ipsf not in pmom:
            pmom[ipsf] = psf_moments(pool[ipsf], conf)
        out['psf_e'][i] = pmom[ipsf][:2]
        out['psf_T'][i] = pmom[ipsf][2]
        out['gal_index'][i] = grec['index']
        out['gal_flux'][i] = grec['flux']
        out['gal_hlr'][i] = grec['hlr']

        if i == 0:
            # compile every numba kernel before anything is timed, so the
            # first object's wall time is not inflated by compilation
            for arm in arms:
                measure(conf, arm, images[0]['obs'], seed)

        for arm in arms:
            n = arm['name']
            for j, im in enumerate(images):
                # same metacal seed for every arm and every image of a galaxy
                res = measure(conf, arm, im['obs'], seed)
                for k in PER_IMAGE + PER_ARM:
                    out[n + '/' + k][i, j] = res[k]

        if verbose and (i + 1) % max(1, nobj // 20) == 0:
            print('%d/%d  %.1f s/object' % (
                i + 1, nobj, (time.time() - t0) / (i + 1)), flush=True)

    out['psf_labels'] = np.array(pool.labels())
    return out


def _layout():
    return [(rot, sign) for rot in ROT_ANGLES for sign in SHEAR_SIGNS]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--chunk', type=int, required=True)
    p.add_argument('--nobj', type=int, required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    with open(args.config) as f:
        conf = yaml.safe_load(f)

    out = run_chunk(conf, args.chunk, args.nobj)
    out['config'] = np.array(yaml.safe_dump(conf))

    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    np.savez_compressed(args.out, **out)


if __name__ == '__main__':
    main()
