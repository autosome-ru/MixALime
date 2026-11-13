# -*- coding: utf-8 -*-
"""
Distributions of the MixALime models. The math behind them is that of JACOBIN.

A count x follows a distribution truncated to x > left. Parameters are those
of the paper: r and p for NB and MCNB, with the concentration k added for
BetaNB. For the binomial models of a count given the coverage, r stands for
the coverage.
"""
from jacobin import NB, BetaNB, MCNB, Binomial, BetaBinomial, TruncatedDistribution, CompoundDist
from functools import lru_cache
import jax.numpy as jnp
import numpy as np
import jax


# Distributions of a count given the count of the other allele.
DISTS = ('NB', 'BetaNB', 'MCNB')
# Distributions of a count given the coverage.
BINOM_DISTS = ('Binom', 'BetaBinom')
# Distributions whose pmf is cheaper as a table over all counts than point by point.
TABLE_DISTS = ('MCNB',)

_BASE = {'NB': NB(), 'BetaNB': BetaNB(), 'MCNB': MCNB(CompoundDist.Binomial), 'Binom': Binomial(),
         'BetaBinom': BetaBinomial()}
_TRUNC = {name: TruncatedDistribution(dist) for name, dist in _BASE.items()}


def transform_r(dist: str, r, p, k, r_transform: str = None):
    """
    Reparametrises r.

    With 'NB' the mean of dist is close to r * p / (1 - p), the mean of NB.
    With 'mean' it is close to r. None leaves r as it is.
    """
    if not r_transform or dist in BINOM_DISTS:
        return r
    if r_transform not in ('NB', 'mean'):
        raise ValueError(f'Unknown transform of r: {r_transform}.')
    if dist == 'NB':
        return r if r_transform == 'NB' else r * (1 - p) / p
    if dist == 'MCNB':
        return r * (1 - p ** r) / ((1 - p) if r_transform == 'NB' else p)
    r = r * ((1 - p) * k - 1) / (p * k)
    return r * p / (1 - p) if r_transform == 'NB' else r


def params(dist: str, r, p, k, r_transform: str = None) -> tuple:
    """Arguments of the JACOBIN distribution for the parameters of the paper."""
    r = transform_r(dist, r, p, k, r_transform)
    if dist == 'BetaNB':
        return r, (1 - p) * k, p * k
    if dist == 'BetaBinom':
        return r, p * k, (1 - p) * k
    return r, p


def logpmf(dist: str, x, r, p, k, r_transform: str = None):
    """Log-pmf of the distribution without truncation."""
    return _BASE[dist].logpmf(x, *params(dist, r, p, k, r_transform))


def _head(dist: str, args: tuple, left: int) -> tuple:
    """
    The counts 0, ..., left and the pmf at them, along a new leading axis.

    The pmf is traced once for all of the counts: a term per count would
    multiply the size of the compiled program by left + 1.
    """
    shape = jnp.broadcast_shapes(*(jnp.shape(t) for t in args))
    i = jnp.arange(int(left) + 1.0).reshape((-1,) + (1,) * len(shape))
    return i, jnp.exp(_BASE[dist].logpmf(i, *args))


def lognorm(dist: str, r, p, k, left: int, r_transform: str = None, stable=False):
    """
    Logarithm of P(X > left), the normalising constant of the truncated distribution.

    It is 1 minus the pmf summed over 0, ..., left, and the difference loses
    its digits once P(X > left) falls below about 1e-6. With stable=True such
    elements take the log-survival function of JACOBIN, at several times the
    compile time of the sum.
    """
    args = params(dist, r, p, k, r_transform)
    _, pmf = _head(dist, args, left)
    head = pmf.sum(axis=0)
    if not stable:
        return jnp.log1p(-head)
    res = jnp.log1p(-jnp.minimum(head, 1.0))
    lost = head > 1.0 - 1e-6

    def tail():
        return jnp.where(lost, jnp.broadcast_to(_BASE[dist].logsf(float(left), *args), res.shape), res)
    return jax.lax.cond(jnp.any(lost), tail, lambda: res)


def logpmf_table(dist: str, r, p, k, left: int, max_sz: int, r_transform: str = None):
    """
    Log-pmf of the truncated distribution at 0, ..., max_sz - 1 for each r.

    r and k are vectors of the same length, p is a scalar. The result has a
    row for each r and is -inf up to left.
    """
    trunc = _TRUNC[dist]

    def row(r, k):
        return trunc.logpmf_recurrent(0, max_sz - 1, max_sz, *params(dist, r, p, k, r_transform), left=int(left) + 1)
    return jax.vmap(row)(r, jnp.broadcast_to(k, jnp.shape(r)))


def mean(dist: str, r, p, k, left: int, r_transform: str = None):
    """Mean of the truncated distribution: that of the whole one without the part at 0, ..., left."""
    args = params(dist, r, p, k, r_transform)
    i, pmf = _head(dist, args, left)
    return (_BASE[dist].mean(*args) - (i * pmf).sum(axis=0)) / (1.0 - pmf.sum(axis=0))


def sf(dist: str, x, r: float, p: float, k: float, left: int, r_transform: str = None) -> np.ndarray:
    """
    P(X > x | X > left) for integers x in any order and scalar parameters.

    The sums run in multiple precision, so a tail far below the double
    precision of 1 - cdf comes out with all of its digits.
    """
    r, p, k = float(r), float(p), float(k)
    args = tuple(float(t) for t in params(dist, r, p, k, r_transform))
    res = _TRUNC[dist].long_sf(np.atleast_1d(np.asarray(x)).astype(int), *args, left=int(left) + 1)
    return np.array([float(t) for t in res])


@lru_cache(maxsize=None)
def _table_fun(max_sz: int, dist: str, r_transform: str):
    """Jitted (r, p, k) -> log-pmf without truncation at 0, ..., max_sz - 1, with a row for each r and k and a scalar p."""
    def row(r, p, k):
        return _BASE[dist].logpmf_recurrent(0, max_sz - 1, max_sz, *params(dist, r, p, k, r_transform))
    return jax.jit(jax.vmap(row, in_axes=(0, None, 0)))


def _sample_tables(dist: str, rows: np.ndarray, left: int, r_transform: str, rng) -> np.ndarray:
    """
    A draw for each (r, p, k) row by inversion of the cdf.

    The cdf comes from a table of the pmf above left with a row per distinct
    parameter row. Each row is scaled by its sum, so a tiny P(X > left) does no harm.
    The table is doubled until its last term is below 1e-13 of the sum.
    """
    uniq, inv = np.unique(rows, axis=0, return_inverse=True)
    inv = inv.ravel()
    order = np.argsort(inv, kind='stable')
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    u = rng.random(len(rows))
    res = np.empty(len(rows), dtype=int)
    for q in np.unique(uniq[:, 1]):
        sel = np.flatnonzero(uniq[:, 1] == q)
        # The number of rows is rounded up so that the tables of different calls share compiled shapes.
        n = 1 << max(4, int(len(sel) - 1).bit_length())
        r, k = (np.pad(uniq[sel, j], (0, n - len(sel)), mode='edge') for j in (0, 2))
        max_sz = 1 << max(8, int(left + 256).bit_length())
        while True:
            lp = np.asarray(_table_fun(max_sz, dist, r_transform)(r, q, k))[:len(sel), int(left) + 1:]
            pmf = np.exp(lp - lp.max(axis=1, keepdims=True))
            cdf = np.cumsum(pmf, axis=1)
            if not np.isfinite(cdf[:, -1]).all():
                raise ValueError(f'The pmf of {dist} is not finite at p = {q}.')
            if (pmf[:, -1] < 1e-13 * cdf[:, -1]).all() or max_sz >= 1 << 22:
                break
            max_sz *= 2
        for row, i in zip(cdf, sel):
            inds = order[bounds[i]:bounds[i + 1]]
            res[inds] = int(left) + 1 + np.searchsorted(row / row[-1], u[inds], side='right')
    return res


def sample(dist: str, r, p, k, left: int, r_transform: str = None, rng=None) -> np.ndarray:
    """
    A draw from the truncated distribution for each element of r, of the shape of r.

    rng is a seed or a numpy Generator. Draws are rejected while at most
    left, and those still rejected after 10 rounds come from the tables.
    """
    rng = np.random.default_rng(rng)
    shape = np.shape(r)
    r, p, k = (np.broadcast_to(np.asarray(t, dtype=float), shape).ravel() for t in (r, p, k))
    rows = np.stack((r, p, k), axis=1)
    if dist in TABLE_DISTS:
        return _sample_tables(dist, rows, left, r_transform, rng).reshape(shape)
    args = params(dist, r, p, k, r_transform)
    base = _BASE[dist]
    res = np.empty(r.shape, dtype=int)
    res[...] = base.sample(r.shape, *args, seed=rng)
    redo = res <= left
    for _ in range(10):
        if not redo.any():
            break
        res[redo] = base.sample(int(redo.sum()), *(a[redo] for a in args), seed=rng)
        redo = res <= left
    if redo.any():
        res[redo] = _sample_tables(dist, rows[redo], left, r_transform, rng)
    return res.reshape(shape)
