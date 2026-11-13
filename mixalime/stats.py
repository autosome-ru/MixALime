# -*- coding: utf-8 -*-
"""Fit indices of a window fit, slice by slice."""
from functools import lru_cache
from scipy.special import logsumexp
import jax.numpy as jnp
import numpy as np
import jax
from .models import modes_logpmf, truncated_logpmf, padded_size, mix, hessian
from .params import SliceParams


def pad(x: np.ndarray, n: int, value: float) -> np.ndarray:
    """x extended to length n with value."""
    return np.concatenate((x, np.full(n - len(x), value)))


def mixture_logpmf(counts: np.ndarray, params: dict, bad: float, left: int, dist: str, r_transform: str, modes=False):
    """
    Log-pmf of each count under the mixture at its slice.

    params holds one value per row for r and, where present, k, w, p1, p2
    (the layout of SliceParams.at). With modes=True the log-pmfs of the two
    components are returned in place of the mixture.
    """
    m = len(counts)
    n = padded_size(m)
    p1 = params.get('p1', np.full(m, bad / (bad + 1)))
    p2 = params.get('p2', np.full(m, 1 / (bad + 1)) if bad != 1 else p1)
    k = params.get('k', np.ones(m))
    args = (pad(counts, n, left + 1.0), pad(params['r'], n, 1.0), pad(k, n, 1.0), pad(p1, n, 0.5), pad(p2, n, 0.5))
    lp1, lp2 = (np.asarray(t)[:m] for t in modes_logpmf(dist, left, r_transform)(*map(jnp.asarray, args)))
    if modes:
        return lp1, lp2
    w = params.get('w', np.ones(m))
    return logsumexp(np.stack((lp1, lp2)), axis=0, b=np.stack((w, 1.0 - w)))


# Bounds of r, w, kappa, p1 and p2.
_BOUNDS = ((0.0, None), (0.0, 1.0), (0.0, None), (0.0, 1.0), (0.0, 1.0))


@lru_cache(maxsize=None)
def _slice_grad(dist: str, left: int, r_transform: str):
    """Jitted gradient of the negative log-likelihood of a slice in (r, w, k, p1, p2)."""
    def nll(theta, counts, weights):
        r, w, k, p1, p2 = theta
        lp = mix(truncated_logpmf(dist, counts, r, p1, k, left, r_transform),
                 truncated_logpmf(dist, counts, r, p2, k, left, r_transform), w)
        return -jnp.where(weights > 0, lp * weights, 0.0).sum()
    return jax.jit(jax.jacfwd(nll))


def fit_indices(data: np.ndarray, params: dict, bad: float, left: int, dist: str, r_transform=None,
                estimate_p=False, adjusted_loglik=False) -> dict:
    """
    Goodness of fit of a window fit at each slice.

    Parameters
    ----------
    data : np.ndarray
        Rows of (count, slice, number of observations) sorted by slice.
    params : dict
        The 'params' entry of a fit.
    bad, left, dist, r_transform, estimate_p
        Settings of the fitted model.
    adjusted_loglik : bool, optional
        Also report the log-likelihood penalised by the log-determinant of
        the observed Fisher information. The default is False.

    Returns
    -------
    dict
        Maps a slice to a dict with the mean log-likelihood 'loglik', the
        number of observations 'n', the G statistic 'stat', its degrees of
        freedom 'dof' and 'rmsea'.
    """
    counts, slices, weights = data.T
    ps = SliceParams(params).at(slices)
    lp = mixture_logpmf(counts, ps, bad, left, dist, r_transform)
    uniq, starts = np.unique(slices, return_index=True)
    two_modes = bad != 1
    # Free parameters of the mixture at a slice: r, w, both p and kappa.
    active = [True, two_modes, dist.startswith('Beta'), estimate_p, estimate_p and two_modes]
    n = np.add.reduceat(weights, starts)
    loglik = np.add.reduceat(lp * weights, starts) / n
    n_rows = np.diff(np.append(starts, len(counts)))
    with np.errstate(divide='ignore', invalid='ignore'):
        stat = 2 * np.add.reduceat(weights * (np.log(weights) - np.log(n[np.searchsorted(uniq, slices)]) - lp), starts)
        dof = n_rows - sum(active)
        rmsea = np.where((n - 1) * dof > 0, np.sqrt(np.maximum(stat - dof, 0) / ((n - 1) * dof)), np.nan)
    res = dict()
    for i, s in enumerate(uniq):
        # Halves of symmetrified counts make n fractional.
        n_i = int(n[i]) if n[i] == int(n[i]) else float(n[i])
        res[int(s)] = {'loglik': float(loglik[i]), 'n': n_i, 'rmsea': float(rmsea[i]), 'stat': float(stat[i]),
                       'dof': int(dof[i])}
    if adjusted_loglik:
        grad = _slice_grad(dist, left, r_transform)
        get = lambda name, default: ps[name] if name in ps else np.full(len(counts), default)
        thetas = np.stack((ps['r'], get('w', 1.0), get('k', 1.0), get('p1', bad / (bad + 1)), get('p2', 1 / (bad + 1))))
        idx = np.flatnonzero(active)
        for i, s in enumerate(uniq):
            a, b = starts[i], starts[i] + n_rows[i]
            m = padded_size(n_rows[i], 64)
            theta = thetas[:, a].astype(float)
            args = (jnp.asarray(pad(counts[a:b], m, left + 1.0)), jnp.asarray(pad(weights[a:b], m, 0.0)))

            def fun(x):
                t = theta.copy()
                t[idx] = x
                return np.asarray(grad(jnp.asarray(t), *args), dtype=float)[idx]
            # The observed Fisher information in the free parameters of the slice.
            fim = hessian(fun, theta[idx], [_BOUNDS[j] for j in idx])
            try:
                eigs = np.linalg.eigvalsh(fim)
                res[int(s)]['loglik_adjusted'] = float(loglik[i] - np.log(eigs[eigs > 1e-12]).sum() / n[i])
            except np.linalg.LinAlgError:
                res[int(s)]['loglik_adjusted'] = float('nan')
    return res
