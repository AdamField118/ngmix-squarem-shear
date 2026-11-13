"""
Image simulation with common random numbers.

For each galaxy we draw four images: shear +g and -g, each at rotations of
0 and 90 degrees.  All four share one noise field and one sub-pixel offset,
and every measurement arm sees the same four images.  Intrinsic shape noise
cancels in the 90-degree pair and noise-induced bias largely cancels in the
+/- pair (Pujol et al. 2019), so differences between arms are measured far
more precisely than any single arm's absolute bias.
"""
import numpy as np
import galsim
import ngmix

ROT_ANGLES = (0.0, 90.0)
SHEAR_SIGNS = (+1, -1)


def make_images(conf, gal, psf_entry, rng):
    """
    draw the four images of one galaxy

    Returns
    -------
    list of dict: keys 'obs', 'sign', 'rot'
    """
    iconf = conf['image']
    scale = iconf['pixel_scale']
    dims = iconf['stamp_size']
    noise = iconf['noise_sigma']
    g = conf['shear']['g']

    psf = psf_entry['obj']
    method = 'no_pixel' if psf_entry['pixel_included'] else 'auto'

    offset = rng.uniform(-0.5, 0.5, size=2)
    noise_im = rng.normal(scale=noise, size=(dims, dims))

    psf_im = psf.drawImage(nx=dims, ny=dims, scale=scale, method=method).array
    psf_im /= psf_im.sum()
    cen = (dims - 1.0) / 2.0
    psf_jac = ngmix.DiagonalJacobian(row=cen, col=cen, scale=scale)
    # psf images are noiseless; the weight only sets the scale ngmix uses
    # to report psf s/n and is ignored by the EM fit
    psf_wt = np.full(psf_im.shape, 1.0 / (1.0e-5 * psf_im.max())**2)

    out = []
    for rot in ROT_ANGLES:
        for sign in SHEAR_SIGNS:
            obj = gal.rotate(rot * galsim.degrees).shear(g1=sign * g, g2=0.0)
            obj = galsim.Convolve(obj, psf)
            im = obj.drawImage(
                nx=dims, ny=dims, scale=scale, method=method,
                offset=offset,
            ).array
            im = im + noise_im
            jac = ngmix.DiagonalJacobian(
                row=cen + offset[1], col=cen + offset[0], scale=scale,
            )
            psf_obs = ngmix.Observation(
                psf_im.copy(), weight=psf_wt.copy(), jacobian=psf_jac,
            )
            obs = ngmix.Observation(
                im, weight=np.full(im.shape, 1.0 / noise**2),
                jacobian=jac, psf=psf_obs,
            )
            out.append({'obs': obs, 'sign': sign, 'rot': rot})
    return out


def psf_moments(psf_entry, conf):
    """
    adaptive-moment ellipticity and size of the native PSF, used to measure
    PSF leakage
    """
    iconf = conf['image']
    scale = iconf['pixel_scale']
    dims = iconf['stamp_size']
    method = 'no_pixel' if psf_entry['pixel_included'] else 'auto'
    im = psf_entry['obj'].drawImage(
        nx=dims, ny=dims, scale=scale, method=method,
    ).array
    cen = (dims - 1.0) / 2.0
    obs = ngmix.Observation(
        im, jacobian=ngmix.DiagonalJacobian(row=cen, col=cen, scale=scale),
    )
    res = ngmix.admom.run_admom(obs, guess=4 * scale**2,
                                rng=np.random.RandomState(1))
    if res['flags'] != 0:
        return np.nan, np.nan, np.nan
    return res['e'][0], res['e'][1], res['T']
