#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .utils import get_init_file, load_pickle, save_pickle, process_pool, closest_bad, dictify_fix
from .models import modes_mean, mode_probs, padded_size, truncated_logpmf, mix, build_windows
from .stats import mixture_logpmf, pad
from .params import SliceParams, nearest, split_name
from . import dists
from multiprocessing import cpu_count
from scipy.optimize import minimize_scalar
from contextlib import nullcontext
from functools import lru_cache
from itertools import product
import jax.numpy as jnp
import numpy as np
import logging
import jax
import os


def _sf(task: tuple) -> np.ndarray:
    return dists.sf(*task)


def tail_probs(tasks: list, n_jobs: int = 1, pool=None) -> list:
    """
    Evaluates dists.sf for each tuple of its arguments in tasks.

    The work is spread over pool, a pool of n_jobs processes, or over a new
    one if pool is None. Processes are started by a forkserver, as a fork of
    a process with a live JAX runtime is not safe.
    """
    chunksize = max(1, len(tasks) // (4 * n_jobs))
    if pool is not None:
        return pool.map(_sf, tasks, chunksize=chunksize)
    if n_jobs <= 1 or len(tasks) < 2:
        return list(map(_sf, tasks))
    with process_pool(min(n_jobs, len(tasks))) as p:
        return p.map(_sf, tasks, chunksize=chunksize)


def _shared_pool(n_jobs: int):
    """A pool of n_jobs processes for all the calls of tail_probs in a run, or a null context for 1 process."""
    return process_pool(n_jobs) if n_jobs > 1 else nullcontext()


def _posterior_weights(w: np.ndarray, lp1: np.ndarray, lp2: np.ndarray, correction: str) -> np.ndarray:
    """
    Weights of the first component after a look at the observation itself.

    'hard' picks the likelier component, 'single' takes the posterior
    probability of the first one. Weights of 0 and 1 are left as they are,
    and so are those of observations impossible under both components.
    """
    res = w.astype(float)
    with np.errstate(invalid='ignore'):
        mixed = (w > 0) & (w < 1) & ~np.isnan(lp1 - lp2)
    if correction == 'hard':
        res[mixed] = lp1[mixed] > lp2[mixed]
    elif correction == 'single':
        with np.errstate(over='ignore', invalid='ignore'):
            odds = w[mixed] / (1 - w[mixed]) * np.exp(lp1[mixed] - lp2[mixed])
            res[mixed] = np.where(np.isinf(odds), 1.0, odds / (odds + 1))
    return res


def _fitted_windows(fit: dict, slices: np.ndarray, sizes: np.ndarray):
    """
    Inclusive indices into slices of the window whose fit gives the estimates at each fitted slice.

    slices and sizes are the fitted slices, ascending, and their numbers of
    observations. The windows are read from the stats or, if the fit does not
    record them, rebuilt from its settings: a run of slices with equal
    estimates shares the window of its first slice. None if neither is possible.
    """
    stats = fit['stats']
    if all('window_lo' in stats[s] for s in slices):
        return (np.searchsorted(slices, [stats[s]['window_lo'] for s in slices]),
                np.searchsorted(slices, [stats[s]['window_hi'] for s in slices]))
    inst = fit['inst_params']
    if 'window_size' not in inst:
        return None
    win = build_windows(sizes, inst['window_size'], inst.get('min_slices', 1), inst.get('window_behavior', 'both'),
                        slices, inst.get('window_ratio'))
    ests = dict()
    for name, value in zip(fit['params']['names'], fit['params']['ests']):
        prefix, s = split_name(name)
        # r and k change from slice to slice within a window.
        if prefix not in (None, 'r', 'k'):
            ests.setdefault(s, dict())[prefix] = value
    first = np.arange(len(slices))
    for i in range(1, len(slices)):
        if ests.get(slices[i]) == ests.get(slices[i - 1]):
            first[i] = first[i - 1]
    return win[first, 0], win[first, 1]


def slice_support(fit: dict, slices, ratio: float) -> np.ndarray:
    """
    Share of the observations behind the estimates at each slice that lie near it.

    The estimates at a slice s are those of the nearest fitted slice, and they
    come from the fit of a window of slices. The share is that of the
    observations of the window at slices from s / ratio to s * ratio. It is 1
    for a fit whose windows are unknown, and for a fit with b and mu fixed, as
    then the line of r is not estimated anywhere.
    """
    slices = np.asarray(slices, dtype=float)
    fixed = dictify_fix(fit.get('inst_params', dict()).get('fix_params') or dict())
    if 'b' in fixed and 'mu' in fixed:
        return np.ones(len(slices))
    stats = fit.get('stats') or dict()
    fitted = np.array(sorted(stats))
    if not len(fitted) or not all('n' in stats[s] for s in fitted):
        return np.ones(len(slices))
    sizes = np.array([stats[s]['n'] for s in fitted], dtype=float)
    windows = _fitted_windows(fit, fitted, sizes)
    if windows is None:
        return np.ones(len(slices))
    i = nearest(fitted, slices)
    a, b = windows[0][i], windows[1][i]
    csum = np.concatenate(([0.0], np.cumsum(sizes)))
    # The near slices of the window are lo..hi, none if hi is lo - 1.
    lo = np.minimum(np.maximum(np.searchsorted(fitted, slices / ratio, 'left'), a), b + 1)
    hi = np.maximum(np.minimum(np.searchsorted(fitted, slices * ratio, 'right') - 1, b), lo - 1)
    return (csum[hi + 1] - csum[lo]) / (csum[b + 1] - csum[a])


def calc_stats(counts: np.ndarray, fit: dict, swap: bool, correction: str = None, gof_tr: float = None,
               dataset_n_thr: float = None, min_support: float = 0.5, support_ratio: float = 2.0,
               unsupported: str = 'conservative', posterior: bool = False, n_jobs: int = 1, pool=None) -> dict:
    """
    P-values and effect sizes of every (ref, alt) pair at one BAD for one allele.

    Parameters
    ----------
    counts : np.ndarray
        Rows of (ref, alt, n).
    fit : dict
        Fit of this BAD and allele: 'params', 'stats' and 'inst_params'.
    swap : bool
        If True, the alt allele is scored given ref, otherwise ref given alt.
    correction : str, optional
        'hard', 'single' or None, see _posterior_weights.
    gof_tr : float, optional
        Slices with RMSEA above it are scored conservatively.
    dataset_n_thr : float, optional
        Slices with fewer observations in the fit are scored conservatively.
    min_support : float, optional
        A slice is unsupported if a smaller share of the observations of the
        window behind its estimates lies within a factor support_ratio of it,
        see slice_support. 0 turns the check off. The default is 0.5.
    support_ratio : float, optional
        The default is 2.
    unsupported : str, optional
        'conservative' scores an unsupported slice conservatively, 'nan' leaves
        it without p-values and effect sizes. The default is 'conservative'.
    posterior : bool, optional
        If True, a pair also gets the posterior probability of the first
        component, the one with the scored allele on more copies. It is the
        weight 'single' gives, whatever the correction. It is nan at BAD 1,
        in a fit with w fixed at 0 or 1, in a conservative slice and where
        the p-value is nan.
    n_jobs : int, optional
        Number of processes for the tail probabilities.
    pool : multiprocessing.pool.Pool, optional
        A pool of n_jobs processes to use in place of a new one.

    Returns
    -------
    dict
        Maps (ref, alt) to (p-value, effect size), or to (p-value, effect
        size, posterior) with posterior. The p-value is the
        probability of a count at least as high as the observed one, the
        effect size is log2 of the count over its expectation. Both are nan
        in a slice with r = 0, where the model has no mass above left, and in
        an unsupported slice with unsupported 'nan'.
    """
    if unsupported not in ('conservative', 'nan'):
        raise ValueError(f"unsupported should be 'conservative' or 'nan', got {unsupported!r}.")
    inst = fit['inst_params']
    bad, left, dist, r_transform = inst['bad'], int(inst['left']), inst['dist'], inst.get('r_transform')
    counts = np.asarray(counts)[:, :2].astype(int)
    main, fixed = (counts[:, 1], counts[:, 0]) if swap else (counts[:, 0], counts[:, 1])
    order = np.lexsort((main, fixed))
    main, fixed = main[order], fixed[order]
    slices, starts, inv = np.unique(fixed, return_index=True, return_inverse=True)
    bounds = np.append(starts, len(fixed))
    ps = SliceParams(fit['params']).at(slices)
    m = len(slices)
    p1, p2 = mode_probs(bad)
    ps.setdefault('w', np.ones(m))
    ps.setdefault('p1', np.full(m, p1))
    ps.setdefault('p2', ps['p1'] if bad == 1 else np.full(m, p2))
    ps.setdefault('k', np.ones(m))
    # In a conservative slice r is the slice itself, all the weight is on the first component and kappa is dropped.
    stats = fit['stats']
    conservative = np.zeros(m, dtype=bool)
    for i, s in enumerate(slices):
        st = stats.get(s, dict())
        if gof_tr is not None and st.get('rmsea', np.inf) > gof_tr:
            conservative[i] = True
        if dataset_n_thr is not None and st.get('n', 0) < dataset_n_thr:
            conservative[i] = True
    weak = np.zeros(m, dtype=bool)
    if min_support:
        weak = slice_support(fit, slices, support_ratio) < min_support
    if weak.any():
        n = int(np.isin(inv, np.flatnonzero(weak)).sum())
        what = 'are scored conservatively' if unsupported == 'conservative' else 'get no p-value'
        logging.warning(f'{n} count pairs at BAD {bad} lie in slices where less than {min_support:g} of the observations'
                        f' behind the fit are within a factor {support_ratio:g} of the slice. They {what}.')
        if unsupported == 'conservative':
            conservative |= weak
    ps['r'] = np.where(conservative, slices, ps['r'])
    ps['w'] = np.where(conservative, 1.0, ps['w'])
    ps['p1'] = np.where(conservative, p1, ps['p1'])
    ps['p2'] = np.where(conservative, p2, ps['p2'])
    cons_dist = 'NB' if dist.startswith('Beta') else dist
    degenerate = ~(ps['r'] > 0)
    if degenerate.any():
        n = int(np.isin(inv, np.flatnonzero(degenerate)).sum())
        logging.warning(f'{n} count pairs at BAD {bad} lie in slices with r = 0 and get no p-value.')
    skip = degenerate | weak if unsupported == 'nan' else degenerate
    pvals = np.full(len(main), np.nan)
    es = np.full(len(main), np.nan)
    post = np.full(len(main), np.nan)
    # A fit with w fixed at 0 or 1 assigns nothing, as in the small dataset strategy 'conservative'.
    assigns = posterior and bad != 1 and dictify_fix(inst.get('fix_params') or dict()).get('w') not in (0, 1)
    for cons, d, sel in ((False, dist, ~conservative & ~skip), (True, cons_dist, conservative & ~skip)):
        if not sel.any():
            continue
        rows = sel[inv]
        row_ps = {name: v[inv][rows] for name, v in ps.items()}
        w = row_ps['w']
        soft = assigns and not cons
        if (correction in ('hard', 'single') or soft) and ((w > 0) & (w < 1)).any():
            lp1, lp2 = mixture_logpmf(main[rows].astype(float), row_ps, bad, left, d,
                                      r_transform, modes=True)
            if soft:
                post[rows] = _posterior_weights(w, lp1, lp2, 'single')
            if correction in ('hard', 'single'):
                w = _posterior_weights(w, lp1, lp2, correction)
        elif soft:
            post[rows] = w
        n = padded_size(m, 64)
        means = modes_mean(d, left, r_transform)(*(pad(ps[t], n, 1.0 if t in ('r', 'k') else 0.5) for t in ('r', 'k', 'p1', 'p2')))
        mean1, mean2 = (np.asarray(t)[:m][inv][rows] for t in means)
        ind = np.where(sel)[0]
        tasks = list()
        for mode in ('p1', 'p2'):
            for i in ind:
                x = main[bounds[i]:bounds[i + 1]] - 1
                tasks.append((d, x, ps['r'][i], ps[mode][i], ps['k'][i], left, r_transform))
            if not (w < 1).any():
                break
        sfs = tail_probs(tasks, n_jobs, pool)
        sf1 = np.concatenate(sfs[:len(ind)])
        sf2 = np.concatenate(sfs[len(ind):]) if len(sfs) > len(ind) else sf1
        pvals[rows] = w * sf1 + (1 - w) * sf2
        # A component of weight 0 may have an infinite mean.
        with np.errstate(invalid='ignore'):
            mean = np.where(w > 0, w * mean1, 0.0) + np.where(w < 1, (1 - w) * mean2, 0.0)
        es[rows] = np.log2(main[rows]) - np.log2(mean)
    main, fixed, pvals, es, post = main.tolist(), fixed.tolist(), pvals.tolist(), es.tolist(), post.tolist()
    vals = zip(pvals, es, post) if posterior else zip(pvals, es)
    if swap:
        return {(f, c): v for c, f, v in zip(main, fixed, vals)}
    return {(c, f): v for c, f, v in zip(main, fixed, vals)}


def assignment_prob(post_ref, post_alt) -> np.ndarray:
    """
    Maximal posterior assignment probability of count pairs.

    post_ref and post_alt are the posterior probabilities of the first
    component in the tests of ref and of alt, the component with the scored
    allele on more copies. The result is the larger of the probabilities that
    both tests put the same allele on more copies. It is near 0 where each
    test puts its allele on more copies, or each on fewer, and nan where a
    posterior is nan.
    """
    post_ref, post_alt = np.asarray(post_ref, dtype=float), np.asarray(post_alt, dtype=float)
    return np.maximum(post_ref * (1 - post_alt), (1 - post_ref) * post_alt)


def _clear_unassigned(ref: dict, alt: dict, bad: float, min_prob: float):
    """
    Drops the posteriors from the results of calc_stats of both alleles at a BAD, in place.

    Both p-values of a pair whose assignment_prob is below min_prob are set to 1.
    """
    pairs = list(ref)
    low = assignment_prob([ref[t][2] for t in pairs], [alt[t][2] for t in pairs]) < min_prob
    if low.any():
        logging.warning(f'{int(low.sum())} count pairs at BAD {bad} have a posterior assignment probability'
                        f' below {min_prob:g}. Both of their p-values are set to 1.')
    for t, drop in zip(pairs, low):
        ref[t] = (1.0 if drop else ref[t][0], ref[t][1])
        alt[t] = (1.0 if drop else alt[t][0], alt[t][1])


def test(name: str, correction: str = None, gof_tr: float = None, dataset_n_thr: float = None,
         fit: str = None, min_support: float = 0.5, support_ratio: float = 2.0, unsupported: str = 'conservative',
         min_posterior_prob: float = 0.0, n_jobs: int = 1):
    """
    Computes p-values and effect sizes for all count pairs of a project.

    fit is a path to a fit file of another project, the fit of the project
    by default. BADs are matched by closest_bad: one missing from the fit of
    the project is skipped with a warning, one missing from another fit
    raises a KeyError. Both p-values of a pair are set to 1 if its
    assignment_prob is below min_posterior_prob, 0 turns this off. n_jobs -1
    stands for all cores, the other arguments are those of calc_stats.
    Returns and saves a dict indexed as [allele][bad][(ref, alt)].
    """
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    fits = load_pickle(fit if fit else f'{name}.fit.{compressor}')
    counts_d = load_pickle(filename)['counts']
    if dataset_n_thr is not None and dataset_n_thr <= 0:
        dataset_n_thr = None
    keys = dict()
    for bad in counts_d:
        try:
            keys[bad] = {allele: closest_bad(fits[allele], bad) for allele in ('ref', 'alt')}
        except KeyError:
            if fit:
                raise KeyError(f'BAD {bad} is present in the data and absent from the fit {fit}.')
            n = int(np.asarray(counts_d[bad])[:, -1].sum())
            logging.warning(f'The fit has no BAD {bad}: its {n} observations are not scored.')
    res = {'ref': dict(), 'alt': dict()}
    with _shared_pool(n_jobs) as pool:
        for bad, key in keys.items():
            for allele in ('ref', 'alt'):
                res[allele][bad] = calc_stats(counts_d[bad], fits[allele][key[allele]], allele != 'ref',
                                              correction=correction, gof_tr=gof_tr, dataset_n_thr=dataset_n_thr,
                                              min_support=min_support, support_ratio=support_ratio,
                                              unsupported=unsupported, posterior=bool(min_posterior_prob),
                                              n_jobs=n_jobs, pool=pool)
            if min_posterior_prob:
                _clear_unassigned(res['ref'][bad], res['alt'][bad], bad, min_posterior_prob)
    filename = f'{name}.comb.{compressor}'
    if os.path.isfile(filename):
        os.remove(filename)
    save_pickle(res, f'{name}.test.{compressor}')
    return res


@lru_cache(maxsize=None)
def _binom_nll(dist: str, left: int, mixture: bool):
    """
    Returns a jitted function (p, k, w, x, n, weights) -> negative
    log-likelihood of the counts x given the coverages n under the
    left-truncated model. With mixture, the components with success
    probabilities p and 1 - p have weights 1 - w and w.
    """
    @jax.jit
    def fun(p, k, w, x, n, weights):
        lp = truncated_logpmf(dist, x, n, p, k, left, None)
        if mixture:
            lp = mix(truncated_logpmf(dist, x, n, 1.0 - p, k, left, None), lp, w)
        return -jnp.where(weights > 0, lp * weights, 0.0).sum()
    return fun


@lru_cache(maxsize=None)
def _binom_mean(dist: str, left: int):
    """Returns a jitted function (n, p, k) -> mean of the left-truncated model at the coverages n."""
    return jax.jit(lambda n, p, k: dists.mean(dist, n, p, k, left))


def binom_modes(w: float, bad: float, p: float = None) -> tuple:
    """
    Success probabilities and weights of the components of the model of a count given the coverage.

    A single component is used if w is None or BAD is 1, its p is 1 / 2 by
    default. Otherwise the components are p, bad / (bad + 1) by default,
    with the weight 1 - w, and 1 - p with the weight w.
    """
    if w is None or bad == 1:
        return (0.5 if p is None else p,), (1.0,)
    p = bad / (bad + 1) if p is None else p
    return (p, 1 - p), (1 - w, w)


def calc_stats_binom(counts: np.ndarray, dist: str, w: float, bad: float, left: int, swap: bool,
                     p: float = None, k: float = None, n_jobs: int = 1, pool=None) -> dict:
    """
    P-values and effect sizes under a left-truncated (beta-)binomial model of
    a count given the coverage.

    Parameters
    ----------
    counts : np.ndarray
        Rows of (ref, alt, n).
    dist : str
        'Binom' or 'BetaBinom'.
    w : float
        Weight of the component with success probability 1 - p. If None or
        if BAD is 1, a single component is used.
    bad : float
        BAD.
    left : int
        Left truncation bound.
    swap : bool
        If True, alt counts are scored.
    p : float, optional
        Success probability. By default it is 1 / 2 for a single component
        and bad / (bad + 1) for a mixture.
    k : float, optional
        Concentration of the beta-binomial distribution.
    n_jobs : int, optional
        Number of processes. The default is 1.
    pool : multiprocessing.pool.Pool, optional
        A pool of n_jobs processes to use in place of a new one.

    Returns
    -------
    dict
        Maps (ref, alt) to (p-value, effect size).
    """
    main, other = (counts[:, 1], counts[:, 0]) if swap else (counts[:, 0], counts[:, 1])
    main, other = main.astype(int), other.astype(int)
    cover = main + other
    order = np.lexsort((main, cover))
    main, other, cover = main[order], other[order], cover[order]
    covers, starts, inv = np.unique(cover, return_index=True, return_inverse=True)
    bounds = np.append(starts, len(cover))
    modes, weights = binom_modes(w, bad, p)
    k = 1.0 if k is None else float(k)
    tasks = [(dist, main[bounds[i]:bounds[i + 1]] - 1, float(n), q, k, left, None)
             for q in modes for i, n in enumerate(covers)]
    sfs = tail_probs(tasks, n_jobs, pool)
    padded = jnp.asarray(pad(covers.astype(float), padded_size(len(covers), 64), 2 * left + 2.0))
    pvals = mean = 0.0
    for i, (q, weight) in enumerate(zip(modes, weights)):
        pvals = pvals + weight * np.concatenate(sfs[i * len(covers):(i + 1) * len(covers)])
        mean = mean + weight * np.asarray(_binom_mean(dist, left)(padded, q, k))[:len(covers)][inv]
    es = np.log2(main) - np.log2(mean)
    main, other, pvals, es = main.tolist(), other.tolist(), pvals.tolist(), es.tolist()
    if swap:
        return {(f, c): (pv, e) for c, f, pv, e in zip(main, other, pvals, es)}
    return {(c, f): (pv, e) for c, f, pv, e in zip(main, other, pvals, es)}


def est_binom_params(counts_d: dict, left: int, w: float, dist: str, est_p=False,
                     max_cover: int = float('inf')) -> dict:
    """
    Maximum likelihood estimates of the parameters of the (beta-)binomial
    model, for each BAD and each allele.

    Each (ref, alt) pair weighs as many observations as it has. Pairs with a
    cover above max_cover are left out. Returns a dict indexed as
    [bad][allele] with (p, k) tuples. p is the default of binom_modes unless
    est_p is set, k is None for the binomial model.
    """
    def est(bad, swap, p, est_k):
        counts = counts_d[bad]
        x, other, n_obs = counts[:, int(swap)], counts[:, 1 - int(swap)], counts[:, 2]
        cover = x + other
        ind = cover <= max_cover
        if not ind.any():
            raise ValueError(f'No observations with a cover of at most {max_cover} at BAD {bad}.')
        size = padded_size(int(ind.sum()))
        args = (pad(x[ind].astype(float), size, left + 1.0), pad(cover[ind].astype(float), size, 2 * left + 2.0),
                pad(n_obs[ind].astype(float), size, 0.0))
        args = tuple(map(jnp.asarray, args))
        mixture = len(binom_modes(w, bad)[0]) > 1
        nll = _binom_nll(dist, left, mixture)
        weight = w if mixture else 0.0
        if est_k:
            fun = lambda k: float(nll(p, k, weight, *args))
        else:
            # The concentration is not known yet when p is estimated.
            fun = lambda q: float(nll(q, 500.0, weight, *args))
        a, b, n = (0.0, 1000.0, 50) if est_k else (0.01, 0.99, 10)
        seps = np.linspace(a, b, n)
        values = np.array([fun(t) for t in seps[1:-1]])
        if not np.isfinite(values).any():
            raise ValueError(f'The likelihood of the {dist} model is not finite at BAD {bad}.')
        i = np.nanargmin(np.where(np.isfinite(values), values, np.nan))
        return float(minimize_scalar(fun, bounds=(seps[i], seps[i + 2]), method='bounded').x)

    res = dict()
    for bad, swap in product(counts_d, (False, True)):
        p = binom_modes(w, bad)[0][0]
        k = None
        if est_p:
            p = est(bad, swap, p, False)
        if dist != 'Binom':
            k = est(bad, swap, p, True)
        res.setdefault(bad, dict())['alt' if swap else 'ref'] = (p, k)
    return res


def binom_test(name: str, w: float, beta=False, estimate_p=False, max_cover=float('inf'), n_jobs=1):
    """
    Scores a project with a left-truncated binomial or beta-binomial model
    of an allelic count given the coverage.

    Returns the p-values and effect sizes indexed as [allele][bad][(ref, alt)],
    saved to the project as well, and the parameters indexed as [bad][allele].
    n_jobs -1 stands for all cores.
    """
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    if max_cover is None:
        max_cover = float('inf')
    if w is not None and not 0 <= w <= 1:
        raise ValueError(f'The weight w should lie between 0 and 1, got {w}.')
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    counts_d = load_pickle(filename)['counts']
    left = int(min(c[:, (0, 1)].min() for c in counts_d.values())) - 1
    dist = 'BetaBinom' if beta else 'Binom'
    params = est_binom_params(counts_d, left, w=w, est_p=estimate_p, dist=dist, max_cover=max_cover)
    res = {'ref': dict(), 'alt': dict()}
    with _shared_pool(n_jobs) as pool:
        for bad in counts_d:
            for allele in ('ref', 'alt'):
                p, k = params[bad][allele]
                res[allele][bad] = calc_stats_binom(counts_d[bad], dist, w, bad, left, allele != 'ref', p=p, k=k,
                                                    n_jobs=n_jobs, pool=pool)
    filename = f'{name}.comb.{compressor}'
    if os.path.isfile(filename):
        os.remove(filename)
    save_pickle(res, f'{name}.test.{compressor}')
    return res, params
