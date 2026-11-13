"""
Metacalibration measurement with interchangeable PSF fitters ("arms").

The set-up follows the SuperBIT pipeline (superbit-lensing,
metacalibration/ngmix_fit.py): a 'gauss' galaxy model fit with the LM fitter,
metacal with psf='dilate' and the PSF of every metacal image fit by an EM
mixture, which the galaxy fit convolves with.  Only the PSF fitter differs
between arms:

    em      ngmix.em.EMFitter with the given tol/maxiter/ngauss and
            accelerate=True/False
    galsim  no PSF model at all: the galaxy is fit with GalsimFitter, which
            convolves with the PSF *image*.  This is the PSF-modelling-error
            free reference.

Every arm is run with a RandomState seeded identically, so metacal noise and
all guesses are shared between arms.
"""
import time
import numpy as np
import ngmix
from ngmix.flags import EM_MAXITER

from .guessers import AnyNGaussPSFGuesser


class CountingEMFitter(ngmix.em.EMFitter):
    """
    EMFitter that records iterations, time and maxiter hits of every fit
    """
    def __init__(self, stats, **kw):
        super().__init__(**kw)
        self.stats = stats

    def go(self, obs, guess, sky=None):
        t0 = time.perf_counter()
        res = super().go(obs=obs, guess=guess, sky=sky)
        self.stats['psf_time'] += time.perf_counter() - t0
        self.stats['psf_nfit'] += 1
        self.stats['psf_numiter'] += res.get('numiter', 0)
        if res['flags'] & EM_MAXITER:
            self.stats['psf_nmaxiter'] += 1
        return res


def new_stats():
    return {'psf_time': 0.0, 'psf_nfit': 0, 'psf_numiter': 0,
            'psf_nmaxiter': 0, 'total_time': 0.0}


def make_bootstrapper(conf, arm, rng, stats):
    mconf = conf['metacal']
    gal_model = mconf.get('gal_model', 'gauss')
    ntry = mconf.get('ntry', 20)
    lm_pars = {'maxfev': 2000, 'xtol': 5.0e-5, 'ftol': 5.0e-5}
    Tguess = mconf.get('Tguess', 4 * conf['image']['pixel_scale']**2)

    kind = arm.get('psf_fitter', 'em')
    if kind == 'em':
        em = arm.get('em', {})
        psf_fitter = CountingEMFitter(
            stats,
            tol=em.get('tol', 1.0e-5),
            maxiter=em.get('maxiter', 500),
            miniter=em.get('miniter', 40),
            accelerate=em.get('accelerate', False),
        )
        psf_guesser = AnyNGaussPSFGuesser(rng=rng, ngauss=arm.get('ngauss', 5))
        psf_runner = ngmix.runners.PSFRunner(
            fitter=psf_fitter, guesser=psf_guesser, ntry=ntry,
        )
        fitter = ngmix.fitting.Fitter(model=gal_model, fit_pars=lm_pars)
        guesser = ngmix.guessers.TPSFFluxGuesser(rng=rng, T=Tguess)
    elif kind == 'galsim':
        psf_runner = None
        fitter = ngmix.fitting.GalsimFitter(
            model=mconf.get('galsim_model', 'gauss'), fit_pars=lm_pars,
        )
        guesser = ngmix.guessers.TFluxGuesser(
            rng=rng, T=Tguess, flux=mconf.get('flux_guess', 100.0),
        )
    else:
        raise ValueError('psf_fitter must be em or galsim')

    runner = ngmix.runners.Runner(fitter=fitter, guesser=guesser, ntry=ntry)
    return ngmix.metacal.MetacalBootstrapper(
        runner=runner, psf_runner=psf_runner, rng=rng,
        psf=mconf.get('psf', 'dilate'),
        step=mconf.get('step', 0.01),
        types=mconf.get('types', ngmix.metacal.METACAL_TYPES),
    )


def measure(conf, arm, obs, seed):
    """
    run metacal on one image with one arm

    Returns
    -------
    dict with g (noshear), R (2x2), Rpsf (2x2 or nan), s2n, T, flags and the
    PSF-fit statistics
    """
    rng = np.random.RandomState(seed)
    stats = new_stats()
    boot = make_bootstrapper(conf, arm, rng, stats)
    step = conf['metacal'].get('step', 0.01)

    out = {'flags': 0, 'g': np.full(2, np.nan), 'R': np.full((2, 2), np.nan),
           'Rpsf': np.full((2, 2), np.nan), 's2n': np.nan, 'T': np.nan}

    t0 = time.perf_counter()
    try:
        resdict, _ = boot.go(obs)
    except Exception:
        out['flags'] = 1
        resdict = None
    stats['total_time'] = time.perf_counter() - t0
    out.update(stats)

    if resdict is None:
        return out
    for key, res in resdict.items():
        if res is None or res['flags'] != 0:
            out['flags'] = 2
            return out

    out['g'] = resdict['noshear']['g'].copy()
    out['s2n'] = resdict['noshear'].get('s2n', np.nan)
    out['T'] = resdict['noshear'].get('T', np.nan)
    for j, c in enumerate('12'):
        gp = resdict[c + 'p']['g']
        gm = resdict[c + 'm']['g']
        out['R'][:, j] = (gp - gm) / (2 * step)
        if c + 'p_psf' in resdict:
            gp = resdict[c + 'p_psf']['g']
            gm = resdict[c + 'm_psf']['g']
            out['Rpsf'][:, j] = (gp - gm) / (2 * step)
    return out
