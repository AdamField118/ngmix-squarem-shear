"""
Galaxy sources.

cosmos    Parametric (Sersic or bulge+disk) fits to HST/ACS COSMOS galaxies
          from the GalSim COSMOS catalog (Mandelbaum et al. 2012; the 25.2
          sample is drawn from the same field as COSMOS2015).  Fluxes are
          rescaled by `flux_scale` and an exposure-dependent zero point is
          left to the noise level in the config.
analytic  Exponential disks with log-normal size and flux distributions and
          Gaussian intrinsic ellipticity, for smoke tests without data.

Each draw returns a GalSim object (before shear and rotation) and a record of
its properties.
"""
import numpy as np
import galsim


class GalaxySource(object):
    def __init__(self, conf):
        self.gconf = conf['galaxies']
        self.kind = self.gconf['source']
        if self.kind == 'cosmos':
            kw = {}
            if 'dir' in self.gconf:
                kw['dir'] = self.gconf['dir']
            self.cat = galsim.COSMOSCatalog(
                file_name=self.gconf.get('file_name', None),
                sample=self.gconf.get('sample', '25.2'),
                exclusion_level=self.gconf.get('exclusion_level', 'marginal'),
                min_hlr=self.gconf.get('min_hlr', 0.0),
                max_hlr=self.gconf.get('max_hlr', 0.0),
                **kw,
            )
        elif self.kind != 'analytic':
            raise ValueError("galaxies source must be 'cosmos' or 'analytic'")

    def draw(self, rng):
        if self.kind == 'cosmos':
            return self._cosmos(rng)
        return self._analytic(rng)

    def _cosmos(self, rng):
        index = rng.randint(self.cat.nobjects)
        gal = self.cat.makeGalaxy(
            index, gal_type='parametric', chromatic=False,
        )
        gal = gal.withFlux(gal.flux * self.gconf.get('flux_scale', 1.0))
        # the size is read from the catalog: gal.calculateHLR() would draw
        # the unconvolved profile, which for an n~6 Sersic needs an FFT of
        # 10^4-10^5 on a side and kills the job
        rec = self.cat.getParametricRecord(index)
        return gal, {'index': index, 'flux': gal.flux,
                     'hlr': float(rec['hlr'][0])}

    def _analytic(self, rng):
        g = self.gconf
        hlr = np.exp(rng.normal(np.log(g.get('hlr_median', 0.4)),
                                g.get('hlr_logsigma', 0.4)))
        flux = np.exp(rng.normal(np.log(g.get('flux_median', 1000.0)),
                                 g.get('flux_logsigma', 0.8)))
        while True:
            e1, e2 = rng.normal(scale=g.get('shape_sigma', 0.2), size=2)
            if e1**2 + e2**2 < 0.8**2:
                break
        gal = galsim.Exponential(half_light_radius=hlr, flux=flux)
        gal = gal.shear(g1=e1, g2=e2)
        return gal, {'index': -1, 'flux': flux, 'hlr': hlr}
