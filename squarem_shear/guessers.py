"""
A starting mixture for any number of gaussians, defined in pixel units.

We do not use ngmix.guessers.GMixPSFGuesser.  It perturbs each component's
centre by up to 0.1 in the jacobian's units, i.e. 0.1 arcsec regardless of
the pixel scale.  That is harmless on 0.141 arcsec SuperBIT pixels, but on
the 0.03 arcsec pixels of the HST/ACS images it is more than the width of
the PSF: a component can start where there is no flux, collapses to zero
size on the first EM step, and every one of the 20 retries fails.  Its
tables also stop at five components.

Here every random perturbation is a fraction of a pixel or of the size
itself, so the guess is the same in pixel units for every instrument.  The
overall flux and size are taken as GMixPSFGuesser does by default (image sum,
and T for a FWHM of 3.5 pixels); component sizes are spread log-uniformly
over a factor `spread` about that T, with weights falling with size.
"""
import numpy as np
import ngmix
from ngmix import moments


class AnyNGaussPSFGuesser(object):
    def __init__(self, rng, ngauss, spread=10.0):
        self.rng = rng
        self.ngauss = ngauss
        self.spread = spread

    def __call__(self, obs):
        rng = self.rng
        scale = obs.jacobian.scale
        flux = obs.image.sum()
        T = moments.fwhm_to_T(3.5 * scale)

        n = self.ngauss
        if n == 1:
            sizes = np.ones(1)
        else:
            sizes = np.geomspace(1.0 / np.sqrt(self.spread),
                                 np.sqrt(self.spread), n)
        weights = 1.0 / sizes
        weights /= weights.sum()

        pars = np.zeros(6 * n)
        for i in range(n):
            Ti = T * sizes[i] * rng.uniform(0.9, 1.1)
            beg = 6 * i
            pars[beg + 0] = flux * weights[i] * rng.uniform(0.9, 1.1)
            # centre within a tenth of a pixel of the stamp centre
            pars[beg + 1] = rng.uniform(-0.1, 0.1) * scale
            pars[beg + 2] = rng.uniform(-0.1, 0.1) * scale
            pars[beg + 3] = Ti / 2 * rng.uniform(0.95, 1.05)
            pars[beg + 4] = rng.uniform(-0.02, 0.02) * Ti
            pars[beg + 5] = Ti / 2 * rng.uniform(0.95, 1.05)
        return ngmix.GMix(pars=pars)
