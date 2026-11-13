# -*- coding: utf-8 -*-
"""Mixture models of allelic read counts and their sliding-window fit."""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from scipy.optimize import minimize
import jax.numpy as jnp
import numpy as np
import logging
import jax
from . import dists


# Layout of a full parameter vector.
PARAMS = ('w', 'b', 'mu', 'b_k', 'mu_k', 'p1', 'p2')
_W, _B, _MU, _BK, _MUK, _P1, _P2 = range(len(PARAMS))


@dataclass
class Parameter:
    name: str
    starts: tuple   # starting points tried when the parameter is estimated
    value: float    # fixed value, None for an estimated parameter
    bound: tuple


def mode_probs(bad: float) -> tuple:
    """Success probabilities of the two mixture components at a BAD."""
    return bad / (bad + 1), 1 / (bad + 1)


def padded_size(n: int, minimum: int = 256, base: int = 2) -> int:
    """Smallest power of base that is at least n and at least minimum."""
    res = 1
    while res < n or res < minimum:
        res *= base
    return res


def symmetrify_counts(data: np.ndarray) -> np.ndarray:
    """
    Makes a table of (ref, alt, n) rows symmetric in ref and alt.

    A pair and its mirror both get half of their summed n, so the total
    number of observations is kept and n may end in .5.
    """
    d = dict()
    for r, a, n in data:
        r, a = int(r), int(a)
        d[(r, a)] = d.get((r, a), 0.0) + n / 2
        d[(a, r)] = d.get((a, r), 0.0) + n / 2
    return np.array([(r, a, d[(r, a)]) for r, a in sorted(d)], dtype=float).reshape(-1, 3)


def build_windows(sizes: np.ndarray, req_size: int, min_slices: int, behavior: str = 'both',
                  slices: np.ndarray = None, ratio: float = None) -> np.ndarray:
    """
    Finds a window of slices around each slice.

    sizes[i] is the number of observations at the i-th slice, slices ascending.
    A window grows from its slice until it holds req_size observations and
    min_slices slices: with behavior 'both' by a slice on the right, then one on
    the left, and so on, with 'right' to the right until the end is reached.
    With ratio, the window of a slice s holds no slice below s / ratio or above
    s * ratio, and is short of the requirements if they are not met there.
    slices are the slice values, needed for ratio only.
    Returns inclusive (first, last) slice indices, one row per slice.
    """
    m = len(sizes)
    csum = np.concatenate(([0], np.cumsum(sizes)))
    res = np.empty((m, 2), dtype=int)
    first, last = np.zeros(m, dtype=int), np.full(m, m - 1)
    if ratio is not None:
        slices = np.asarray(slices, dtype=float)
        first = np.searchsorted(slices, slices / ratio, 'left')
        last = np.searchsorted(slices, slices * ratio, 'right') - 1

    def enough(a, b):
        return csum[b + 1] - csum[a] >= req_size and b - a + 1 >= min_slices

    for i in range(m):
        f, t = first[i], last[i]
        if behavior == 'right':
            lo, hi = i, t
            while lo < hi:
                mid = (lo + hi) // 2
                if enough(i, mid):
                    hi = mid
                else:
                    lo = mid + 1
            b = lo
            lo, hi = f, i
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if enough(mid, b):
                    lo = mid
                else:
                    hi = mid - 1
            res[i] = lo, b
            continue
        # The q-th state of the growth is [i - q // 2, i + (q + 1) // 2] cut to the
        # available slices. Both the size and the width grow with q.
        lo, hi = 0, 2 * max(i - f, t - i)
        while lo < hi:
            mid = (lo + hi) // 2
            if enough(max(i - mid // 2, f), min(i + (mid + 1) // 2, t)):
                hi = mid
            else:
                lo = mid + 1
        res[i] = max(i - lo // 2, f), min(i + (lo + 1) // 2, t)
    return res


def truncated_logpmf(dist: str, x, r, p, k, left: int, r_transform: str, stable=False):
    """Log-pmf of a distribution truncated to x > left. stable is that of dists.lognorm."""
    lp = dists.logpmf(dist, x, r, p, k, r_transform) - dists.lognorm(dist, r, p, k, left, r_transform, stable)
    return jnp.where(x > left, lp, -jnp.inf)


@lru_cache(maxsize=None)
def modes_logpmf(dist: str, left: int, r_transform: str):
    """
    Returns a jitted function (counts, r, k, p1, p2) -> (lp1, lp2) with the
    log-pmfs of both mixture components. All arguments are arrays of one shape.
    """
    @jax.jit
    def fun(counts, r, k, p1, p2):
        return (truncated_logpmf(dist, counts, r, p1, k, left, r_transform),
                truncated_logpmf(dist, counts, r, p2, k, left, r_transform))
    return fun


@lru_cache(maxsize=None)
def modes_mean(dist: str, left: int, r_transform: str):
    """Returns a jitted function (r, k, p1, p2) -> (mean1, mean2) for both mixture components."""
    @jax.jit
    def fun(r, k, p1, p2):
        return (dists.mean(dist, r, p1, k, left, r_transform),
                dists.mean(dist, r, p2, k, left, r_transform))
    return fun


def mix(lp1, lp2, w):
    """
    Log of w * exp(lp1) + (1 - w) * exp(lp2).

    The larger of the two terms is taken out of the logarithm, so the
    derivative in w is exact at w = 0 and w = 1 as well.
    """
    d = lp1 - lp2
    # Both terms are -inf at a count below the truncation bound.
    d = jnp.where(jnp.isnan(d), 0.0, d)
    e = jnp.exp(-jnp.abs(d))
    top = jnp.where(d > 0, lp1, lp2)
    t = jnp.where(d > 0, w + (1.0 - w) * e, 1.0 - w + w * e)
    # t is zero when the larger term has zero weight and the ratio of the terms underflows.
    return jnp.where(t > 0, top + jnp.log(jnp.where(t > 0, t, 1.0)), top - jnp.abs(d))


def hessian(grad, x: np.ndarray, bounds=None, step: float = 1e-4) -> np.ndarray:
    """
    Hessian of a function at x by differences of its gradient.

    grad maps a point to the gradient there, bounds is a list of (lower, upper)
    with None for no bound. The differences are central, and one-sided of the
    same order where a central step would cross a bound. A second derivative
    of the compiled likelihood takes far longer to compile than the gradient.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    res = np.empty((n, n))
    g0 = None
    for i in range(n):
        h = step * max(1.0, abs(x[i]))
        lo, hi = bounds[i] if bounds is not None else (None, None)
        e = np.zeros(n)
        e[i] = h
        side = 0
        if lo is not None and x[i] - h < lo:
            side = 1
        elif hi is not None and x[i] + h > hi:
            side = -1
        if not side:
            res[i] = (grad(x + e) - grad(x - e)) / (2 * h)
            continue
        if g0 is None:
            g0 = grad(x)
        res[i] = side * (-3 * g0 + 4 * grad(x + side * e) - grad(x + 2 * side * e)) / (2 * h)
    return (res + res.T) / 2


@lru_cache(maxsize=None)
def _objective(dist: str, left: int, r_transform: str, two_modes: bool, active: tuple, prior: tuple):
    """
    Jitted objective of a window: the mean negative log-likelihood minus the log-prior of kappa, if any.

    Maps (x, fixed, pmask, slc, *data) to the value and its gradient in x. x holds the parameters of active,
    fixed is a full parameter vector, pmask multiplies x and slc is the centre slice of the window. data is
    (counts, index, slices, weights) with index the position of the slice of a row in slices, or (slices,
    weights) with weights of shape (slices, counts) for a distribution computed as a table.
    """
    idx = np.array(active, dtype=int)
    table = dist in dists.TABLE_DISTS

    def logprob(theta, slices, counts, index, max_sz):
        # r at slice 0 is mu alone, and at r = 0 the log-pmf of a count of 0 is nan.
        r = jnp.maximum(theta[_B] * slices + theta[_MU], 1e-12)
        k = theta[_BK] * jnp.log(jnp.maximum(slices, 1.0)) + theta[_MUK]
        if table:
            pmf = lambda p: dists.logpmf_table(dist, r, p, k, left, max_sz, r_transform)
        else:
            # The normalising constant of the truncation is the same for all rows of a slice.
            pmf = lambda p: (dists.logpmf(dist, counts, r[index], p, k[index], r_transform)
                             - dists.lognorm(dist, r, p, k, left, r_transform)[index])
        if not two_modes:
            return pmf(theta[_P1])
        # Both modes go through one trace of the pmf, a trace per mode would double the compiled program.
        lp = jax.vmap(pmf)(jnp.stack((theta[_P1], theta[_P2])))
        return mix(lp[0], lp[1], theta[_W])

    def nll_sum(x, fixed, pmask, slc, *data):
        theta = fixed.at[idx].set(x * pmask) if len(idx) else fixed
        weights = data[-1]
        if table:
            lp = logprob(theta, data[0], None, None, weights.shape[1])
        else:
            lp = logprob(theta, data[2], data[0], data[1], None)
        # A log-pmf is at most 0, so +inf comes from a normaliser lost to rounding and counts as impossible.
        # A nan stays, so that the optimizer rejects the step: a finite penalty misleads its line search.
        lp = jnp.where(jnp.isnan(lp) | jnp.isfinite(lp), lp, -1e19)
        res = -jnp.where(weights > 0, lp * weights, 0.0).sum()
        if prior is not None:
            alpha, by_n, by_slice, kind = prior
            if by_n:
                alpha = alpha * weights.sum()
            if by_slice:
                alpha = alpha * slc
            kappa = theta[_MUK]
            res += alpha / kappa ** (2 if kind == 'normal' else 1) + 2 * jnp.log(kappa)
        return res

    def value_and_grad(x, *args):
        n = args[-1].sum()

        def fun(x):
            v = nll_sum(x, *args) / n
            return v, v
        grad, value = jax.jacfwd(fun, has_aux=True)(x)
        return value, grad

    return jax.jit(value_and_grad)


class ModelWindow():
    def __init__(self, bad: float, left: int, dist='NB', window_size=10000, min_slices=3,
                 window_behavior='both', estimate_p=False, fix_params=None, left_k=1,
                 regul_alpha=0.0, regul_n=True, regul_slice=True, regul_prior='laplace',
                 r_transform=None, symmetrify=False, kappa_right=None, adjust_line=False, window_ratio=None):
        """
        A mixture of two left-truncated distributions fitted in a sliding window.

        The counts of one allele are modelled given the count of the other
        allele (a slice). Inside a window of slices r depends on the slice as
        r = b * slice + mu, kappa of the Beta distributions is constant.

        Parameters
        ----------
        bad : float
            Background Allelic Dosage.
        left : int
            Left truncation bound, counts are greater than left.
        dist : str, optional
            One of dists.DISTS. The default is 'NB'.
        window_size : int, optional
            Minimal number of observations in a window. The default is 10000.
        min_slices : int, optional
            Minimal number of slices in a window. The default is 3.
        window_behavior : str, optional
            'both' or 'right', see build_windows. The default is 'both'.
        estimate_p : bool, optional
            If True, p1 and p2 are estimated, otherwise they are fixed to
            bad / (bad + 1) and 1 / (bad + 1). The default is False.
        fix_params : dict, optional
            Mapping of parameter names to the values they are fixed to.
        left_k : float, optional
            Lower bound of kappa. For BetaNB with r_transform and a fixed p it
            is at least BAD + 1.001, so that the transformed r is positive. The
            default is 1.
        regul_alpha : float, optional
            Strength of the prior on kappa, 0 disables it. The default is 0.0.
        regul_n : bool, optional
            Multiply alpha by the number of observations in a window.
        regul_slice : bool, optional
            Multiply alpha by the slice at the centre of a window.
        regul_prior : str, optional
            'laplace' penalises 1 / kappa, 'normal' penalises 1 / kappa ** 2.
        r_transform : str, optional
            Reparametrisation of r, see dists.transform_r.
        symmetrify : bool, optional
            Symmetrify the counts before the fit. The default is False.
        kappa_right : float, optional
            Upper bound of kappa. The default is None.
        adjust_line : bool, optional
            Pull b and mu of each window towards those of the previous window
            as far as r in the window allows. The default is False.
        window_ratio : float, optional
            The window of a slice s holds no slice below s / window_ratio or
            above s * window_ratio, see build_windows. The default is None,
            for no such bound.
        """
        if dist not in dists.DISTS:
            raise ValueError(f'Unknown distribution {dist}.')
        self.bad = bad
        self.left = int(left)
        self.dist = dist
        self.req_size = window_size
        self.min_slices = min_slices
        self.window_behavior = window_behavior
        if window_ratio is not None and not window_ratio > 1:
            raise ValueError(f'window_ratio should be above 1, got {window_ratio}.')
        self.window_ratio = window_ratio
        self.estimate_p = estimate_p
        self.r_transform = r_transform
        self.symmetrify = symmetrify
        self.adjust_line = adjust_line
        self.two_modes = bad != 1
        beta = dist.startswith('Beta')
        p1, p2 = mode_probs(bad)
        k_low = left_k
        if dist == 'BetaNB' and r_transform and not estimate_p:
            # The transform of r takes a = (1 - p) * kappa above 1 in both modes.
            k_low = max(k_low, bad + 1 + 1e-3)
        if beta and kappa_right is not None and kappa_right <= k_low:
            raise ValueError(f'The upper bound of kappa {kappa_right} is not above its lower bound {k_low}.')
        k_starts = (k_low + 1, max(1, k_low + 26.0))
        if kappa_right is not None:
            k_starts = tuple(dict.fromkeys(min(t, kappa_right) for t in k_starts))
        k_bound = (k_low, kappa_right)
        params = [
            Parameter('w', (0.5,), None if self.two_modes else 1.0, (0.0, 1.0)),
            Parameter('b', (1.0, 0.25), None, (0.05, None)),
            Parameter('mu', (0.0,), None, (0.0, None)),
            Parameter('b_k', (0.0,), 0.0, (0.0, None)),
            Parameter('mu_k', k_starts, None if beta else 0.0, k_bound),
            Parameter('p1', (p1,), None if estimate_p else p1, (0.01, 0.99)),
            Parameter('p2', (p2,), None if estimate_p and self.two_modes else p2, (0.01, 0.99)),
            ]
        unknown = set(fix_params or ()) - set(PARAMS)
        if unknown:
            raise ValueError(f'Unknown parameters to fix: {", ".join(sorted(unknown))}. The parameters are {", ".join(PARAMS)}.')
        for p in params:
            if fix_params and p.name in fix_params:
                p.value = float(fix_params[p.name])
        self.active = tuple(i for i, p in enumerate(params) if p.value is None)
        self.fixed = np.array([0.0 if p.value is None else p.value for p in params])
        self.bounds = [params[i].bound for i in self.active]
        self.x0 = [np.array(t, dtype=float) for t in product(*(params[i].starts for i in self.active))]
        # Parameters reported per slice. Those of an absent mode or of a distribution without kappa are left out.
        self.names = [p.name for p in params if (beta or p.name not in ('b_k', 'mu_k')) and (self.two_modes or p.name != 'p2')]
        prior = None
        if regul_alpha and _MUK in self.active:
            prior = (float(regul_alpha), bool(regul_n), bool(regul_slice), regul_prior)
        # A prior scaled by the slice makes the fit of a window depend on the slice it is built for.
        self.reuse_windows = prior is None or not regul_slice
        self._value_and_grad = _objective(dist, self.left, r_transform, self.two_modes, self.active, prior)

    def load_dataset(self, data: np.ndarray):
        """
        Sorts (count, slice, n) rows by slice and builds the windows.

        Rows with a count or a slice not above the truncation bound are dropped.
        """
        data = np.asarray(data)
        data = data[(data[:, 0] > self.left) & (data[:, 1] > self.left)]
        if not len(data):
            raise ValueError('No observations above the left truncation bound.')
        data = data[np.lexsort((data[:, 0], data[:, 1]))].astype(float)
        self.data = data
        self.slices, starts = np.unique(data[:, 1], return_index=True)
        self.slices = self.slices.astype(int)
        # Rows of the i-th slice are data[row_bounds[i]:row_bounds[i + 1]].
        self.row_bounds = np.append(starts, len(data))
        self.slice_sizes = np.add.reduceat(data[:, 2], starts)
        self.windows = build_windows(self.slice_sizes, self.req_size, self.min_slices, self.window_behavior,
                                     self.slices, self.window_ratio)

    def _window_data(self, a: int, b: int) -> tuple:
        """
        Padded arrays of the rows of slices a..b in the layout the objective takes.

        The objective is compiled once per shape of these arrays. The sizes
        are rounded up so that the windows of a dataset share a few shapes.
        """
        rows = self.data[self.row_bounds[a]:self.row_bounds[b + 1]]
        table = self.dist in dists.TABLE_DISTS
        # A padded slice costs a row of the table, and next to nothing in the other layout.
        slices = np.full(padded_size(b - a + 1, 16) if table else padded_size(b - a + 1, 64, 4), self.left + 1.0)
        slices[:b - a + 1] = self.slices[a:b + 1]
        index = np.searchsorted(self.slices[a:b + 1], rows[:, 1])
        if table:
            weights = np.zeros((len(slices), padded_size(rows[:, 0].max() + 1, 256)))
            weights[index, rows[:, 0].astype(int)] = rows[:, 2]
            return jnp.asarray(slices), jnp.asarray(weights)
        # The gradient reads some 150 arrays of rows at once, and arrays of a power of 2 rows put the same row of
        # each into one cache set. 8 rows more take several times less time.
        n = padded_size(len(rows), 2048) + 8
        counts = np.full(n, self.left + 1.0)
        weights = np.zeros(n)
        counts[:len(rows)], weights[:len(rows)] = rows[:, 0], rows[:, 2]
        return jnp.asarray(counts), jnp.asarray(np.pad(index, (0, n - len(rows)))), jnp.asarray(slices), jnp.asarray(weights)

    def prepare_start(self, start: np.ndarray, standoff=0.04, eps=0.01) -> np.ndarray:
        """Moves a starting point away from the bounds."""
        start = start.copy()
        for i, ((a, b), v) in enumerate(zip(self.bounds, start)):
            if a is not None and b is not None:
                # Capped, as 4% of a (0, kappa_right) range is far above a typical kappa.
                d = min((b - a) * standoff, standoff)
                start[i] = min(max(v, a + d), b - d)
            elif a is not None:
                start[i] = max(v, a + eps)
            elif b is not None:
                start[i] = min(v, b - eps)
        return start

    def _minimize(self, fun, x0s: list, optimizer: str, prev, only_prev: bool):
        """
        Runs an optimizer from each starting point and returns the best result.

        With only_prev the search stops at the first converged run that
        stays within an order of magnitude of prev.
        """
        options = {'maxiter': 1000, 'disp': False, 'ftol': 1e-12}
        if optimizer == 'L-BFGS-B':
            options['gtol'] = 1e-8
        best = None
        for i, x0 in enumerate(x0s):
            logging.debug(f'Fitting model to data: starting point {i}/{len(x0s)}.')
            r = minimize(fun, x0, jac=True, method=optimizer, bounds=self.bounds, options=options)
            if best is None and np.isfinite(r.fun) or best is not None and r.fun < best.fun:
                best = r
                if r.success and only_prev:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        shift = np.abs((r.x - prev) / prev).max()
                    if shift < 10.0:
                        break
        return best

    def _fit_window(self, a: int, b: int, slc: int, pmask: np.ndarray, prev, first: bool, calc_std: bool):
        """
        Fits the parameters to the rows of slices a..b.

        prev is the result of the previous window or None. The first window
        is fitted from every starting point by both optimizers. Returns a
        scipy result with the mean log-likelihood in loglik and, on request,
        the standard errors in std, or None if no run gave a finite likelihood.
        """
        data = self._window_data(a, b)
        args = (jnp.asarray(self.fixed), jnp.asarray(pmask), float(slc), *data)

        def fun(x):
            value, grad = self._value_and_grad(jnp.asarray(x), *args)
            return float(value), np.asarray(grad, dtype=float)

        if not self.active:
            x = np.zeros(0)
            best = _Result(x=x, fun=fun(x)[0], success=True)
        else:
            x0s = list(self.x0)
            start = None
            if prev is not None:
                start = self.prepare_start(prev.x)
                x0s.insert(0, start)
            only_prev = prev is not None and bool(prev.success) and not first
            other = 'L-BFGS-B' if self.optimizer == 'SLSQP' else 'SLSQP'
            best = self._minimize(fun, x0s, self.optimizer, start, only_prev)
            if best is None or first:
                r = self._minimize(fun, x0s, other, start, only_prev)
                if r is not None and (best is None or r.fun < best.fun):
                    best = r
            if best is None:
                logging.debug("Optimizer didn't converge.")
                return None
        best.loglik = -float(best.fun)
        best.cov = np.zeros((len(best.x), len(best.x)))
        if calc_std and self.active:
            # The observed Fisher information is the Hessian of the summed objective.
            fim = hessian(lambda x: fun(x)[1], best.x, self.bounds) * float(data[-1].sum())
            try:
                best.cov = np.linalg.pinv(fim) * np.outer(pmask, pmask)
            except np.linalg.LinAlgError:
                best.cov = np.eye(len(best.x))
        best.std = np.sqrt(np.abs(np.diag(best.cov)))
        # A masked parameter has no effect on the likelihood and is reported as zero.
        best.x = best.x * pmask
        return best

    def _adjust_line(self, cur: np.ndarray, prev: np.ndarray, slices: np.ndarray, eps=1e-1):
        """
        Finds (b, mu) closest to prev whose line stays within eps per slice,
        in the summed squares, of the line of cur over the given slices.
        """
        target = cur[0] * slices + cur[1]
        con = lambda p: eps * len(slices) - np.sum((p[0] * slices + p[1] - target) ** 2)
        return minimize(lambda p: np.sum((p - prev) ** 2), prev, constraints=[{'type': 'ineq', 'fun': con}])

    def fit(self, data: np.ndarray, calc_std=False, optimizer='SLSQP', stop_slice_n=10) -> dict:
        """
        Fits the model to (count, slice, n) rows window by window.

        Parameters
        ----------
        data : np.ndarray
            Rows of (count of the modelled allele, count of the other allele,
            number of observations), unique in the first two columns.
        calc_std : bool, optional
            Compute standard errors of the estimates. The default is False.
        optimizer : str, optional
            'SLSQP' or 'L-BFGS-B'. The other one is the fallback. The default is 'SLSQP'.
        stop_slice_n : int, optional
            A slice with fewer observations takes the last fitted window. The
            default is 10.

        Returns
        -------
        dict
            Per-slice estimates under keys like b5, mu5, loglik5, success5,
            fixed parameters included. With calc_std, r5 and k5 hold r and
            kappa at the slice, and 'std' maps the same keys to standard errors.
            The window whose fit gives the estimates at the i-th slice is
            self.sources[i], as inclusive slice indices.
        """
        self.optimizer = optimizer
        if self.symmetrify:
            data = symmetrify_counts(data)
        self.load_dataset(data)
        ests = dict()
        stds = dict()
        pmask = np.ones(len(self.active))
        i_mu = self.active.index(_MU) if _MU in self.active else None
        i_b = self.active.index(_B) if _B in self.active else None
        prev = prev_window = None
        self.sources = np.empty((len(self.slices), 2), dtype=int)
        for i, (s, (a, b)) in enumerate(zip(self.slices, self.windows)):
            low_coverage = self.slice_sizes[i] < stop_slice_n
            if i_mu is not None and i_b is not None:
                # b and mu are not identifiable in a window of a single slice.
                pmask[i_mu] = float(b > a)
            if prev is None or not (low_coverage or self.reuse_windows and prev_window == (a, b)):
                r = self._fit_window(a, b, s, pmask, prev, i == 0, calc_std)
                if r is None:
                    logging.warning(f'Total optimization failure at slice {s}. That is likely due to very low coverage.'
                                    ' Using the previous fit.')
                    if prev is None:
                        raise RuntimeError(f'Optimization failed at the first slice ({s}), no fit to fall back to.')
                else:
                    if self.adjust_line and prev is not None and i_b is not None and i_mu is not None:
                        ind = [i_b, i_mu]
                        adj = self._adjust_line(r.x[ind], prev.x[ind], self.slices[a:b + 1].astype(float))
                        if adj.success:
                            r.x[ind] = adj.x
                    prev, prev_window = r, (a, b)
            self.sources[i] = prev_window
            theta = self.fixed.copy()
            theta[list(self.active)] = prev.x
            std = np.zeros(len(PARAMS))
            std[list(self.active)] = prev.std
            for name in self.names:
                j = PARAMS.index(name)
                ests[f'{name}{s}'] = float(theta[j])
                stds[f'{name}{s}'] = float(std[j])
            ests[f'loglik{s}'] = prev.loglik
            ests[f'success{s}'] = float(prev.success)
            stds[f'loglik{s}'] = stds[f'success{s}'] = float('nan')
            if calc_std:
                # r and k are lines in the slice and in its log, so their errors need the covariances of the lines.
                cov = np.zeros((len(PARAMS), len(PARAMS)))
                cov[np.ix_(self.active, self.active)] = prev.cov
                lines = [('r', _B, _MU, s)] + ([('k', _BK, _MUK, np.log(max(s, 1)))] if 'mu_k' in self.names else [])
                for name, i, j, t in lines:
                    ests[f'{name}{s}'] = float(theta[i] * t + theta[j])
                    stds[f'{name}{s}'] = float(np.sqrt(abs(t * t * cov[i, i] + 2 * t * cov[i, j] + cov[j, j])))
        if calc_std:
            ests['std'] = stds
        self.params = ests
        return ests


class _Result():
    """A stand-in for a scipy optimization result of a model without estimated parameters."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
