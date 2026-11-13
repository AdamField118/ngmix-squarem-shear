"""
A starting mixture for any number of gaussians.

ngmix.guessers.GMixPSFGuesser has tables for at most five components.  For
larger mixtures we take the starting flux and T the same way it does, and
spread the component sizes log-uniformly over a factor `spread` about T,
with weights falling with size and random perturbations of every parameter.
For ngauss <= 5 the ngmix guesser is used unchanged.
"""
import numpy as np
import ngmix
from ngmix import moments


class AnyNGaussPSFGuesser(object):
    def __init__(self, rng, ngauss, spread=10.0):
        self.rng = rng
        self.ngauss = ngauss
        self.spread = spread
        if ngauss <= 5:
            self._ngmix = ngmix.guessers.GMixPSFGuesser(rng=rng, ngauss=ngauss)
        else:
            self._ngmix = None

    def __call__(self, obs):
        if self._ngmix is not None:
            return self._ngmix(obs=obs)

        rng = self.rng
        scale = obs.jacobian.scale
        flux = obs.image.sum()
        T = moments.fwhm_to_T(3.5 * scale)

        n = self.ngauss
        sizes = np.geomspace(1.0 / np.sqrt(self.spread),
                             np.sqrt(self.spread), n)
        weights = 1.0 / sizes
        weights /= weights.sum()

        pars = np.zeros(6 * n)
        for i in range(n):
            Ti = T * sizes[i] * rng.uniform(0.9, 1.1)
            beg = 6 * i
            pars[beg + 0] = flux * weights[i] * rng.uniform(0.9, 1.1)
            pars[beg + 1] = rng.uniform(-0.1, 0.1) * scale
            pars[beg + 2] = rng.uniform(-0.1, 0.1) * scale
            pars[beg + 3] = Ti / 2 * rng.uniform(0.95, 1.05)
            pars[beg + 4] = rng.uniform(-0.02, 0.02) * Ti
            pars[beg + 5] = Ti / 2 * rng.uniform(0.95, 1.05)
        return ngmix.GMix(pars=pars)
