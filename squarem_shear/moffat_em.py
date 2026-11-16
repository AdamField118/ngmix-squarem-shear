"""
EM, optionally SQUAREM-accelerated, for a mixture of 2D Moffat profiles.

A 2D Moffat profile I(r) ~ (1 + r^2/a^2)^-beta is exactly a bivariate
Student-t density with nu = 2 beta - 2 degrees of freedom,

    f(x) = 1 / (2 pi |S|^1/2) * (1 + delta/nu)^-(nu/2 + 1),
    delta = (x - mu)^T S^-1 (x - mu),

so a mixture of Moffats is a Student-t mixture, which has an EM algorithm
(McLachlan & Peel 2000; Peel & McLachlan 2000).  For a pixelized image d_i
(normalized, used as weights) the steps are

    E:  tau_ik = pi_k f_k(x_i) / q(x_i)                 responsibilities
        u_ik   = (nu_k + 2) / (nu_k + delta_ik)          latent precision
    M:  pi_k   = sum_i d_i tau_ik
        mu_k   = sum d tau u x / sum d tau u
        S_k    = sum d tau u (x - mu_k)(x - mu_k)^T / sum d tau
        nu_k   : root of  -psi(nu/2) + log(nu/2) + 1
                          + sum d tau (log u - u) / sum d tau
                          + psi((nu_old + 2)/2) - log((nu_old + 2)/2) = 0

(the nu step is an ECM conditional maximization).  With u = 1 and no nu
step this is Gaussian-mixture EM, which the module also provides
(gaussian=True) so that per-iteration costs can be compared within one
implementation.

Parameters are a flat array, 7 per component:
    pi, mu_u, mu_v, S_uu, S_uv, S_vv, log nu
Like ngmix's M step, the M step here is projected onto valid mixtures: a
component's weight is floored at 1e-12, and the smaller eigenvalue of its
covariance at `varmin`, by default the variance of the pixel response,
pixel^2/12.  Because the model is evaluated only at pixel centres, nothing
else stops a component from collapsing onto a single pixel or a single row
of pixels, where it buys likelihood without describing the PSF.

The SQUAREM runner matches ngmix's em_run_squarem: SqS3 step, a stabilizing
EM step, rejection of extrapolations that are not valid mixtures, and a
stop on the fractional change in the log likelihood across the two plain EM
steps that begin each cycle.  The log likelihood here is the observed-data
one, sum_i d_i log q(x_i), not ngmix's elogL.

Positions are the pixel centres in the observation's jacobian units and
the model is evaluated at those centres, as ngmix's EM does.
"""
import time
import numpy as np
from numba import njit

NPAR = 7
LOG_NU_MIN = np.log(0.2)
LOG_NU_MAX = np.log(1000.0)


@njit(cache=True)
def _digamma(x):
    r = 0.0
    while x < 6.0:
        r -= 1.0 / x
        x += 1.0
    f = 1.0 / (x * x)
    series = 1.0 / 12 - f * (1.0 / 120 - f * (1.0 / 252 - f * (
        1.0 / 240 - f / 132)))
    return r + np.log(x) - 0.5 / x - f * series


@njit(cache=True)
def _floor_eig(a, b, c, vmin):
    """raise the eigenvalues of [[a, b], [b, c]] to at least vmin"""
    tr = 0.5 * (a + c)
    disc = np.sqrt(max(0.25 * (a - c) ** 2 + b * b, 0.0))
    l1, l2 = tr + disc, tr - disc
    if l2 >= vmin:
        return a, b, c
    l1, l2 = max(l1, vmin), vmin
    if disc == 0.0:
        return l1, 0.0, l1
    # eigenvector of l1 is (cos t, sin t) with tan 2t = 2b / (a - c)
    t = 0.5 * np.arctan2(2 * b, a - c)
    ct, st = np.cos(t), np.sin(t)
    return (l1 * ct * ct + l2 * st * st, (l1 - l2) * ct * st,
            l1 * st * st + l2 * ct * ct)


@njit(cache=True)
def _nu_eq(lnu, c):
    """the ECM equation for nu, as a function of log nu"""
    nu = np.exp(lnu)
    return -_digamma(0.5 * nu) + np.log(0.5 * nu) + c


@njit(cache=True)
def _valid(p, ncomp):
    for k in range(ncomp):
        b = k * NPAR
        for j in range(NPAR):
            if not np.isfinite(p[b + j]):
                return False
        if not (p[b] > 0 and p[b + 3] > 0 and p[b + 5] > 0):
            return False
        if p[b + 3] * p[b + 5] - p[b + 4] * p[b + 4] <= 1e-300:
            return False
    return True


@njit(cache=True)
def _sweep(p, out, u, v, d, ncomp, gaussian, varmin):
    """one E and M step from p into out; returns the log likelihood at p"""
    npix = d.size
    tau = np.zeros((ncomp, npix))
    uw = np.ones((ncomp, npix))
    loglike = 0.0
    for i in range(npix):
        q = 0.0
        for k in range(ncomp):
            b = k * NPAR
            suu, suv, svv = p[b + 3], p[b + 4], p[b + 5]
            det = suu * svv - suv * suv
            du, dv = u[i] - p[b + 1], v[i] - p[b + 2]
            delta = (svv * du * du - 2 * suv * du * dv + suu * dv * dv) / det
            if gaussian:
                f = np.exp(-0.5 * delta) / (2 * np.pi * np.sqrt(det))
            else:
                nu = np.exp(p[b + 6])
                f = (1.0 + delta / nu) ** (-(0.5 * nu + 1.0)) / (
                    2 * np.pi * np.sqrt(det))
                uw[k, i] = (nu + 2.0) / (nu + delta)
            tau[k, i] = p[b] * f
            q += tau[k, i]
        if q > 0:
            loglike += d[i] * np.log(q)
            for k in range(ncomp):
                tau[k, i] /= q
        else:
            for k in range(ncomp):
                tau[k, i] = 0.0

    for k in range(ncomp):
        b = k * NPAR
        sw = 0.0
        swu = 0.0
        su = 0.0
        sv = 0.0
        slog = 0.0
        for i in range(npix):
            w = d[i] * tau[k, i]
            sw += w
            wu = w * uw[k, i]
            swu += wu
            su += wu * u[i]
            sv += wu * v[i]
            if not gaussian:
                slog += w * (np.log(uw[k, i]) - uw[k, i])
        out[b] = max(sw, 1e-12)
        if sw <= 1e-12 or swu <= 1e-300:
            for j in range(1, NPAR):
                out[b + j] = p[b + j]
            continue
        mu_u, mu_v = su / swu, sv / swu
        suu = 0.0
        suv = 0.0
        svv = 0.0
        for i in range(npix):
            wu = d[i] * tau[k, i] * uw[k, i]
            du, dv = u[i] - mu_u, v[i] - mu_v
            suu += wu * du * du
            suv += wu * du * dv
            svv += wu * dv * dv
        suu, suv, svv = _floor_eig(suu / sw, suv / sw, svv / sw, varmin)
        out[b + 1], out[b + 2] = mu_u, mu_v
        out[b + 3], out[b + 4], out[b + 5] = suu, suv, svv
        if gaussian:
            out[b + 6] = p[b + 6]
            continue
        # nu: bisection in log nu on the ECM equation
        nu_old = np.exp(p[b + 6])
        c = 1.0 + slog / sw + _digamma(0.5 * (nu_old + 2)) - np.log(
            0.5 * (nu_old + 2))
        lo, hi = LOG_NU_MIN, LOG_NU_MAX
        glo, ghi = _nu_eq(lo, c), _nu_eq(hi, c)
        if glo * ghi > 0:
            out[b + 6] = hi if ghi > 0 else lo
        else:
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                gm = _nu_eq(mid, c)
                if gm * glo > 0:
                    lo, glo = mid, gm
                else:
                    hi = mid
            out[b + 6] = 0.5 * (lo + hi)
    return loglike


@njit(cache=True)
def run(p0, u, v, d, ncomp, gaussian, tol, maxiter, miniter, accelerate,
        varmin):
    """
    Returns (params, numiter, loglike, flags); flags 1 = maxiter reached,
    2 = no valid mixture.
    """
    npar = p0.size
    p = p0.copy()
    p1 = np.zeros(npar)
    p2 = np.zeros(npar)
    pnew = np.zeros(npar)
    tmp = np.zeros(npar)
    numiter = 0
    step_max = 1.0
    ll = 0.0
    while numiter < maxiter:
        ll0 = _sweep(p, p1, u, v, d, ncomp, gaussian, varmin)
        numiter += 1
        if not _valid(p1, ncomp):
            return p, numiter, ll0, 2
        if numiter >= maxiter:
            p[:] = p1
            break
        ll1 = _sweep(p1, p2, u, v, d, ncomp, gaussian, varmin)
        numiter += 1
        if not _valid(p2, ncomp):
            return p1, numiter, ll1, 2
        ll = ll1
        if numiter >= miniter and ll1 != 0 and abs((ll1 - ll0) / ll1) < tol:
            p[:] = p2
            return p, numiter, ll, 0
        if not accelerate:
            p[:] = p2
            continue
        sr2 = 0.0
        sv2 = 0.0
        for j in range(npar):
            r = p1[j] - p[j]
            vv = p2[j] - 2 * p1[j] + p[j]
            sr2 += r * r
            sv2 += vv * vv
        if sr2 <= 0 or sv2 <= 0 or numiter >= maxiter:
            p[:] = p2
            continue
        s = np.sqrt(sr2 / sv2)
        s = min(max(s, 1.0), step_max)
        for j in range(npar):
            r = p1[j] - p[j]
            vv = p2[j] - 2 * p1[j] + p[j]
            pnew[j] = p[j] + 2 * s * r + s * s * vv
        if _valid(pnew, ncomp):
            _sweep(pnew, tmp, u, v, d, ncomp, gaussian, varmin)
            numiter += 1
            if _valid(tmp, ncomp):
                p[:] = tmp
            else:
                p[:] = p2
        else:
            p[:] = p2
            if s == step_max:
                step_max = max(1.0, step_max / 4.0)
            s = 1.0
        if s == step_max:
            step_max *= 4.0
    return p, numiter, ll, 1


def pixels(obs):
    """pixel-centre coordinates (jacobian units) and normalized weights"""
    im = obs.image
    nrow, ncol = im.shape
    row, col = np.mgrid[0:nrow, 0:ncol]
    v, u = obs.jacobian.get_vu(row.ravel().astype(float),
                               col.ravel().astype(float))
    d = np.clip(im.ravel().astype(float), 0, None)
    return np.asarray(u, float), np.asarray(v, float), d / d.sum()


def initial(obs, ncomp, rng, nu0=4.0, spread=6.0):
    """
    a starting mixture in pixel units, like squarem_shear.guessers: sizes
    log-spaced over `spread` about a 3.5-pixel FWHM, small perturbations
    """
    scale = obs.jacobian.scale
    s2 = (3.5 * scale / 2.3548) ** 2
    sizes = (np.ones(1) if ncomp == 1 else
             np.geomspace(1 / np.sqrt(spread), np.sqrt(spread), ncomp))
    w = 1.0 / sizes
    w /= w.sum()
    p = np.zeros(ncomp * NPAR)
    for k in range(ncomp):
        b = k * NPAR
        sk = s2 * sizes[k] * rng.uniform(0.9, 1.1)
        p[b] = w[k] * rng.uniform(0.9, 1.1)
        p[b + 1:b + 3] = rng.uniform(-0.1, 0.1, 2) * scale
        p[b + 3] = sk * rng.uniform(0.95, 1.05)
        p[b + 4] = sk * rng.uniform(-0.02, 0.02)
        p[b + 5] = sk * rng.uniform(0.95, 1.05)
        p[b + 6] = np.log(nu0)
    p[0::NPAR] /= p[0::NPAR].sum()
    return p


def fit(obs, ncomp, gaussian=False, tol=1e-6, maxiter=50000, miniter=10,
        accelerate=True, seed=0):
    """
    fit a Moffat (or Gaussian) mixture to a PSF observation

    Returns dict with params, numiter, loglike, flags, time and the model
    image (normalized to the image flux).
    """
    # timed from the observation, like EMFitter.go
    t0 = time.perf_counter()
    u, v, d = pixels(obs)
    p0 = initial(obs, ncomp, np.random.RandomState(seed))
    varmin = obs.jacobian.scale ** 2 / 12.0   # the pixel response
    p, n, ll, flags = run(p0, u, v, d, ncomp, gaussian, tol, maxiter,
                          miniter, accelerate, varmin)
    dt = time.perf_counter() - t0
    model = model_image(p, u, v, ncomp, gaussian).reshape(obs.image.shape)
    model *= obs.image.sum() / model.sum()
    return {'params': p, 'numiter': n, 'loglike': ll, 'flags': flags,
            'time': dt, 'model': model}


def model_image(p, u, v, ncomp, gaussian):
    q = np.zeros(u.size)
    for k in range(ncomp):
        b = k * NPAR
        suu, suv, svv = p[b + 3], p[b + 4], p[b + 5]
        det = suu * svv - suv * suv
        du, dv = u - p[b + 1], v - p[b + 2]
        delta = (svv * du * du - 2 * suv * du * dv + suu * dv * dv) / det
        if gaussian:
            f = np.exp(-0.5 * delta)
        else:
            nu = np.exp(p[b + 6])
            f = (1.0 + delta / nu) ** (-(0.5 * nu + 1.0))
        q += p[b] * f / (2 * np.pi * np.sqrt(det))
    return q


def beta(p, ncomp):
    """Moffat beta = nu/2 + 1 of each component"""
    return np.exp(p[6::NPAR][:ncomp]) / 2 + 1
