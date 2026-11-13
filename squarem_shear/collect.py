"""
Combine chunks and estimate m, c, R and PSF leakage for every arm, with
paired bootstrap errors on the differences between arms.

    python -m squarem_shear.collect results/roman/chunk*.npz \
        --out results/roman/summary.json --reference galsim

Estimators (per galaxy k, averaging its four images; g is the input shear):

    D_k  = mean over rotations of [g1(+g) - g1(-g)] / 2
    S_k  = mean over all four images of (g1, g2)
    R_k  = mean over all four images of diag(R)

    m     = <D> / (g <R11>) - 1
    c_i   = <S_i> / <R_ii>
    alpha : least-squares slope of S_k/<R> against the native PSF
            ellipticity e_psf,k, both components stacked

A galaxy is kept only if every image succeeded in every arm, so all arms are
evaluated on identical galaxies.  An optional cut on the *true* flux
(`selection: {min_true_flux: ...}` in the config) removes the faintest
objects; it cannot depend on the shear, so it adds no selection bias.
"""
import argparse
import json
import numpy as np


def load(files):
    data = None
    for f in sorted(files):
        d = dict(np.load(f, allow_pickle=False))
        if data is None:
            data = {k: [v] for k, v in d.items()}
        else:
            for k, v in d.items():
                data[k].append(v)
    out = {}
    for k, v in data.items():
        if k in ('sign', 'rot', 'config', 'psf_labels'):
            out[k] = v[0]
        else:
            out[k] = np.concatenate(v)
    return out


def arm_names(data):
    return sorted({k.split('/')[0] for k in data if '/' in k})


def per_galaxy(data, arm, sign, rot):
    g = data[arm + '/g']
    R = data[arm + '/R']
    plus = sign > 0
    D = np.zeros(g.shape[0])
    for r in np.unique(rot):
        ip = np.where(plus & (rot == r))[0][0]
        im = np.where(~plus & (rot == r))[0][0]
        D += (g[:, ip, 0] - g[:, im, 0]) / 2
    D /= len(np.unique(rot))
    S = g.mean(axis=1)
    Rd = np.stack([R[:, :, 0, 0].mean(axis=1), R[:, :, 1, 1].mean(axis=1)],
                  axis=1)
    return D, S, Rd


def estimate(D, S, Rd, epsf, gtrue):
    R11, R22 = Rd[:, 0].mean(), Rd[:, 1].mean()
    m = D.mean() / (gtrue * R11) - 1
    c1 = S[:, 0].mean() / R11
    c2 = S[:, 1].mean() / R22
    y = np.concatenate([S[:, 0] / R11 - c1, S[:, 1] / R22 - c2])
    x = np.concatenate([epsf[:, 0] - epsf[:, 0].mean(),
                        epsf[:, 1] - epsf[:, 1].mean()])
    alpha = (x * y).sum() / (x * x).sum() if (x * x).sum() > 0 else np.nan
    return {'m': m, 'c1': c1, 'c2': c2, 'R11': R11, 'R22': R22,
            'alpha': alpha}


def summarize(data, gtrue, reference=None, nboot=1000, seed=7,
              min_true_flux=0.0):
    arms = arm_names(data)
    sign, rot = data['sign'], data['rot']

    ok = data['gal_flux'] >= min_true_flux
    for a in arms:
        ok &= np.all(data[a + '/flags'] == 0, axis=1)
    epsf = data['psf_e'][ok]
    ok_psf = np.all(np.isfinite(epsf), axis=1)

    pg = {a: [x[ok] for x in per_galaxy(data, a, sign, rot)] for a in arms}

    point = {a: estimate(*pg[a], epsf, gtrue) for a in arms}
    rng = np.random.RandomState(seed)
    n = ok.sum()
    boots = {a: [] for a in arms}
    for _ in range(nboot):
        idx = rng.randint(n, size=n)
        idx = idx[ok_psf[idx]]
        for a in arms:
            D, S, Rd = pg[a]
            boots[a].append(estimate(D[idx], S[idx], Rd[idx], epsf[idx],
                                     gtrue))

    keys = list(point[arms[0]].keys())
    s2n = data[arms[0] + '/s2n'][ok].ravel()
    summary = {'ngal': int(n), 'nfail': int((~ok).sum()), 'arms': {},
               # galaxies replaced before drawing because GalSim would
               # have needed an FFT larger than image.max_fft_size
               'nredraw': int(data['gal_nredraw'].sum())
               if 'gal_nredraw' in data else 0,
               's2n_quantiles': {str(q): float(np.nanquantile(s2n, q))
                                 for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    for a in arms:
        s = {}
        for k in keys:
            b = np.array([x[k] for x in boots[a]])
            s[k] = point[a][k]
            s[k + '_err'] = float(np.std(b))
            if reference is not None and a != reference:
                bref = np.array([x[k] for x in boots[reference]])
                s['d' + k] = point[a][k] - point[reference][k]
                s['d' + k + '_err'] = float(np.std(b - bref))
        s.update(speed(data, a, ok))
        summary['arms'][a] = {k: float(v) for k, v in s.items()}
    return summary


def _stats(prefix, x):
    """mean, median, std and 5/95th percentiles of a per-object quantity"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        prefix + '_mean': x.mean(),
        prefix + '_median': np.median(x),
        prefix + '_std': x.std(ddof=1) if x.size > 1 else 0.0,
        prefix + '_p05': np.percentile(x, 5),
        prefix + '_p95': np.percentile(x, 95),
        prefix + '_n': float(x.size),
    }


def speed(data, arm, ok):
    """
    cost of an arm, per object (summed over its four images, i.e. four
    metacal measurements of nine PSF fits each): wall time, the fraction of
    it spent in EM PSF fits, EM iterations per PSF fit and the fraction of
    PSF fits that hit maxiter.  Each as mean/median/std/percentiles over
    objects.
    """
    nf = data[arm + '/psf_nfit'][ok].sum(axis=1)
    if nf.sum() == 0:
        return {}
    psf_time = data[arm + '/psf_time'][ok].sum(axis=1)
    tot_time = data[arm + '/total_time'][ok].sum(axis=1)
    numiter = data[arm + '/psf_numiter'][ok].sum(axis=1)
    nmax = data[arm + '/psf_nmaxiter'][ok].sum(axis=1)
    has = nf > 0
    out = {}
    out.update(_stats('time_per_object', tot_time))
    out.update(_stats('psf_time_frac', psf_time[has] / tot_time[has]))
    out.update(_stats('psf_iter_per_fit', numiter[has] / nf[has]))
    out['psf_maxiter_frac'] = nmax.sum() / nf.sum()
    out['psf_time_frac_total'] = psf_time.sum() / tot_time.sum()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('files', nargs='+')
    p.add_argument('--out', required=True)
    p.add_argument('--reference', default=None)
    p.add_argument('--nboot', type=int, default=1000)
    args = p.parse_args()

    data = load(args.files)
    import yaml
    conf = yaml.safe_load(str(data['config']))
    sel = conf.get('selection', {})
    s = summarize(data, conf['shear']['g'], reference=args.reference,
                  nboot=args.nboot,
                  min_true_flux=sel.get('min_true_flux', 0.0))
    s['config'] = conf
    with open(args.out, 'w') as f:
        json.dump(s, f, indent=1)
    print('kept %d galaxies, %d dropped, %d redrawn for FFT size; '
          's/n quantiles %s' % (s['ngal'], s['nfail'], s['nredraw'],
                                s['s2n_quantiles']))
    for a, v in s['arms'].items():
        print('%-18s m=%+.2e+-%.1e c1=%+.1e c2=%+.1e alpha=%+.3f R11=%.4f '
              'psf_frac=%.2f iter/fit=%.0f' % (
                  a, v['m'], v['m_err'], v['c1'], v['c2'], v['alpha'],
                  v['R11'], v.get('psf_time_frac_mean', 0),
                  v.get('psf_iter_per_fit_mean', 0)))


if __name__ == '__main__':
    main()
