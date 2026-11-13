"""
The three PSF sources used in the paper, each returned as a pool of
achromatic GalSim objects in arcsec.

Every source builds its pool from a fixed seed, so all jobs of a run see the
same pool and an object's PSF is fully identified by its pool index.

Sources
-------
superbit  PSFEx models (.psf files) from SuperBIT exposures.  Positions are
          drawn uniformly over each model's fitted range (POLZERO +/-
          POLSCAL/2 in the file header), and the pool cycles through the files
          so every model is used equally often.
cosmos    The HST/ACS F814W PSF images distributed with the GalSim COSMOS
          real-galaxy catalog.
roman     galsim.roman.getPSF: the Roman design pupil (struts, obscuration)
          and field-dependent design aberrations, at random SCA positions.

Each pool entry is a dict with the galsim object, `pixel_included` (whether
the detector pixel response is already in the model, true of PSFEx and of
the ACS images; such models are drawn with method='no_pixel') and a label.

The pool is lazy: positions and indices are fixed when it is built, but a
PSF is only constructed the first time it is used, so a job that touches a
few PSFs out of a pool of thousands does not pay for the rest.
"""
import glob
import numpy as np
import galsim


def build_pool(conf):
    """
    build the PSF pool described by conf['psf']

    Returns
    -------
    list of dict with keys 'obj', 'pixel_included', 'label'
    """
    pconf = conf['psf']
    source = pconf['source']
    rng = np.random.RandomState(pconf.get('seed', 31415))
    npool = pconf.get('npool', 50)

    builders = {
        'superbit': _psfex,
        'psfex': _psfex,
        'cosmos': _cosmos,
        'roman': _roman,
    }
    if source not in builders:
        raise ValueError('psf source must be one of %s' % list(builders))

    return LazyPool(builders[source](pconf, rng, npool))


class LazyPool(object):
    """
    list-like pool built from (label, pixel_included, make) specs
    """
    def __init__(self, specs):
        self.specs = specs
        self.cache = {}

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, i):
        if i not in self.cache:
            label, pixel_included, make = self.specs[i]
            self.cache[i] = {'obj': make(), 'pixel_included': pixel_included,
                             'label': label}
        return self.cache[i]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def labels(self):
        return [s[0] for s in self.specs]


def psfex_range(fname):
    """
    the (xmin, xmax, ymin, ymax) range over which a PSFEx model was fit
    """
    from astropy.io import fits
    h = fits.getheader(fname, ext=1)
    out = []
    for i in (1, 2):
        zero, scal = h['POLZERO%d' % i], h['POLSCAL%d' % i]
        out += [zero - scal / 2, zero + scal / 2]
    return out


def _psfex(pconf, rng, npool):
    import galsim.des

    files = sorted(glob.glob(pconf['files']))
    if len(files) == 0:
        raise IOError('no psfex files match %s' % pconf['files'])
    native_scale = pconf['native_pixel_scale']

    ranges = [psfex_range(f) for f in files]
    models = {}

    def make(k, x, y):
        if k not in models:
            models[k] = galsim.des.DES_PSFEx(files[k])
        # PSFEx models are in pixel units; convert to arcsec preserving flux
        return models[k].getPSF(galsim.PositionD(x, y)).dilate(native_scale)

    specs = []
    for i in range(npool):
        k = i % len(files)
        xmin, xmax, ymin, ymax = ranges[k]
        x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        specs.append(('%s (%.0f,%.0f)' % (files[k], x, y), True,
                      lambda k=k, x=x, y=y: make(k, x, y)))
    return specs


def _cosmos(pconf, rng, npool):
    kw = {}
    if 'dir' in pconf:
        kw['dir'] = pconf['dir']
    rgc = galsim.RealGalaxyCatalog(
        pconf.get('file_name', None), sample=pconf.get('sample', '25.2'), **kw,
    )
    indices = rng.choice(rgc.nobjects, size=npool, replace=False)
    return [('cosmos acs psf %d' % i, True,
             lambda i=int(i): rgc.getPSF(i)) for i in indices]


def _roman(pconf, rng, npool):
    import galsim.roman as roman

    band = pconf.get('band', 'H158')
    bandpasses = roman.getBandpasses()
    wavelength = pconf.get('wavelength', bandpasses[band].effective_wavelength)
    scas = pconf.get('scas', list(range(1, roman.n_sca + 1)))

    pupil_bin = pconf.get('pupil_bin', 4)

    def make(sca, x, y):
        return roman.getPSF(
            sca, band, SCA_pos=galsim.PositionD(x, y),
            wavelength=wavelength, pupil_bin=pupil_bin,
        )

    specs = []
    for _ in range(npool):
        sca = int(rng.choice(scas))
        x, y = rng.uniform(1, roman.n_pix, size=2)
        specs.append(('roman %s SCA%02d (%d,%d)' % (band, sca, x, y), False,
                      lambda sca=sca, x=x, y=y: make(sca, x, y)))
    return specs
