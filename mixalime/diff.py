#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Differential allele-specificity tests between groups of samples."""
from .utils import get_init_file, select_filenames, load_pickle, save_pickle, process_pool
from .models import truncated_logpmf, padded_size, symmetrify_counts
from .params import SliceParams
from .stats import pad
from . import dists
from multiprocessing import cpu_count
from statsmodels.stats import multitest
from collections import defaultdict
from dataclasses import dataclass
from scipy.stats import chi2, norm
from functools import partial, lru_cache
from typing import List, Tuple
from scipy.optimize import minimize_scalar, OptimizeResult
import jax.numpy as jnp
import pandas as pd
import numpy as np
import logging
import zlib
import jax
import re
import os


@lru_cache(maxsize=None)
def _likelihood(dist: str, left: int, r_transform: str):
    """
    Jitted functions of p for rows of (counts, r, k, weights).

    Returns the negative log-likelihood and the weighted score of each row.
    The normaliser is the stable one, as an estimate of p may come close to
    0 or 1, and its second derivative takes minutes to compile.
    """
    def rows(p, counts, r, k, weights):
        lp = truncated_logpmf(dist, counts, r, p, k, left, r_transform, stable=True)
        return jnp.where(weights > 0, lp * weights, 0.0)

    def nll(p, counts, r, k, weights):
        return -rows(p, counts, r, k, weights).sum()
    return jax.jit(nll), jax.jit(jax.jacfwd(rows))


@lru_cache(maxsize=None)
def _moments(dist: str, left: int, r_transform: str):
    """Jitted sums of pmf * log2(x), pmf * log2(x)^2, pmf * x and pmf over a grid x for each (r, k)."""
    def fun(p, r, k, x):
        pr = jnp.exp(truncated_logpmf(dist, x[None, :], r[:, None], p, k[:, None], left, r_transform, stable=True))
        lx = jnp.log2(x)[None, :]
        return (pr * lx).sum(axis=1), (pr * lx ** 2).sum(axis=1), (pr * x[None, :]).sum(axis=1), pr.sum(axis=1)
    return jax.jit(fun)


@dataclass
class GroupFit:
    """
    Estimate of p for a table of counts.

    Holds p, the negative log-likelihood at it, whether the optimizer
    converged, the number of observations n, the padded data args and the
    number of rows m before padding.
    """
    p: float
    nll: float
    success: bool
    n: float
    args: tuple
    m: int


class GroupModel():
    """
    Model of the counts of one allele at one BAD for a group of samples.

    A count given the count of the other allele follows a single left-truncated
    distribution. Its r and kappa come from a window fit and p is estimated.
    """

    def __init__(self, fit: dict, param_mode='window'):
        """fit is the entry of a fit file for one allele and one BAD."""
        inst = fit['inst_params']
        self.dist = inst['dist']
        self.left = int(inst['left'])
        self.r_transform = inst.get('r_transform')
        self.symmetrify = bool(inst.get('symmetrify', False))
        self.params = SliceParams(fit['params'])
        self.param_mode = param_mode
        if self.dist.startswith('Beta') and 'mu_k' not in self.params:
            raise KeyError('No concentration parameter in the fit.')

    def line(self, fixed: np.ndarray, weights: np.ndarray) -> tuple:
        """
        r and kappa for each count of the other allele in fixed.

        In the window mode each count takes the parameters of its window. In
        the line mode all counts take those of the window at their mean.
        """
        ref = None if self.param_mode == 'window' else np.average(fixed, weights=weights)
        ps = self.params.at(fixed, ref=ref)
        return ps['r'], ps.get('k', np.ones(len(fixed)))

    def prepare(self, data: np.ndarray, symmetrify=None) -> tuple:
        """Turns a table of (count, other count, n) rows into padded (counts, r, k, weights) and the number of rows."""
        if self.symmetrify if symmetrify is None else symmetrify:
            data = symmetrify_counts(data)
        counts, fixed, weights = (np.asarray(t, dtype=float) for t in np.asarray(data).T)
        r, k = self.line(fixed, weights)
        n = padded_size(len(counts), 8)
        cols = ((counts, self.left + 1.0), (r, 1.0), (k, 1.0), (weights, 0.0))
        return tuple(jnp.asarray(pad(t, n, v)) for t, v in cols), len(counts)

    def estimate(self, args: tuple, steps=10) -> OptimizeResult:
        """Maximum likelihood estimate of p: a grid search followed by Brent between the neighbours of the best node."""
        nll = _likelihood(self.dist, self.left, self.r_transform)[0]

        def f(p):
            v = float(nll(p, *args))
            return v if np.isfinite(v) else np.inf
        grid = np.linspace(0.01, 0.99, steps)
        values = np.array([f(p) for p in grid])
        if not np.isfinite(values).any():
            return OptimizeResult(x=float('nan'), fun=float('nan'), success=False)
        i = int(np.argmin(values))
        bounds = (grid[i - 1] if i else 0.0, grid[i + 1] if i + 1 < steps else 1.0)
        res = minimize_scalar(f, bounds=bounds, method='bounded', options={'xatol': 1e-16, 'maxiter': 200})
        res.x = float(res.x)
        return res

    def sample(self, data: np.ndarray, p: float, rng) -> np.ndarray:
        """
        Draws counts under p for the rows of data and returns a table of the same layout.

        A row of weight n gets ceil(n) draws of weight n / ceil(n) each, so the
        halves of a symmetrified table keep their share.
        """
        counts, fixed, weights = (np.asarray(t, dtype=float) for t in np.asarray(data).T)
        r, k = self.line(fixed, weights)
        n = np.ceil(weights).astype(int)
        draws = dists.sample(self.dist, np.repeat(r, n), p, np.repeat(k, n), self.left, self.r_transform, rng)
        rows, inv = np.unique(np.stack((draws, np.repeat(fixed, n)), axis=1), axis=0, return_inverse=True)
        return np.column_stack((rows, np.bincount(inv.ravel(), weights=np.repeat(weights / n, n))))

    def fit(self, data: np.ndarray, n_bootstrap=0, rng=None) -> GroupFit:
        """
        Estimates p from a table of (count, other count, n) rows.

        With n_bootstrap > 0 the estimate is corrected for bias by a
        parametric bootstrap of that many samples, drawn with rng (a seed or a
        numpy Generator). A fit whose likelihood is not finite at the estimate
        is a failure.
        """
        if self.symmetrify:
            data = symmetrify_counts(data)
        args, m = self.prepare(data, symmetrify=False)
        n = float(np.asarray(data)[:, -1].sum())
        res = self.estimate(args)
        if not np.isfinite(res.x):
            return GroupFit(float('nan'), float('nan'), False, n, args, m)
        p = res.x
        if n_bootstrap:
            rng = np.random.default_rng(rng)
            bs = list()
            for _ in range(n_bootstrap):
                r = self.estimate(self.prepare(self.sample(data, p, rng), symmetrify=False)[0])
                if r.success and np.isfinite(r.x):
                    bs.append(r.x)
            if bs:
                p = 2 * p - np.mean(bs)
        p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
        nll = self.nll(args, p)
        return GroupFit(p, nll, bool(res.success) and np.isfinite(nll), n, args, m)

    def nll(self, args: tuple, p: float) -> float:
        return float(_likelihood(self.dist, self.left, self.r_transform)[0](p, *args))

    def variance(self, fit: GroupFit, robust=True) -> float:
        """
        Variance of the estimate of p, nan if the likelihood is not concave at it.

        With robust=True it is the sandwich estimate scaled by n / (n - 1).
        The score of a table with a single distinct row vanishes at the
        estimate, so for such a table the inverse Fisher information is
        returned. The curvature is a central difference of the score.
        """
        score = _likelihood(self.dist, self.left, self.r_transform)[1]
        step = 1e-4 * min(fit.p, 1.0 - fit.p)
        total = lambda p: float(np.asarray(score(p, *fit.args)).sum())
        h = (total(fit.p - step) - total(fit.p + step)) / (2 * step)
        if not np.isfinite(h) or h <= 0:
            return float('nan')
        if not robust or fit.m < 2 or fit.n < 2:
            return 1.0 / h
        g, w = np.asarray(score(fit.p, *fit.args))[:fit.m], np.asarray(fit.args[-1])[:fit.m]
        return float((g ** 2 / w).sum() / h ** 2 * fit.n / (fit.n - 1))

    def _moments(self, fit: GroupFit, p: float, tol=1e-9, max_size=1 << 17) -> tuple:
        """Sums of _moments for the rows of fit over a grid doubled until it holds all but tol of the mass."""
        _, r, k, _ = fit.args
        fun = _moments(self.dist, self.left, self.r_transform)
        size = 256
        while True:
            x = jnp.arange(self.left + 1.0, self.left + 1.0 + size)
            res = [np.asarray(t)[:fit.m] for t in fun(p, r, k, x)]
            if res[-1].min() > 1.0 - tol or size >= max_size:
                return res
            size *= 2

    def log_moments(self, fit: GroupFit, p: float) -> tuple:
        """Mean and variance of log2 of a count under p, averaged over the observations of fit."""
        m1, m2, _, _ = self._moments(fit, p)
        w = np.asarray(fit.args[-1])[:fit.m]
        return float((w * m1).sum() / fit.n), float((w * (m2 - m1 ** 2)).sum() / fit.n)

    def log_mean(self, fit: GroupFit, p: float) -> float:
        """log2 of the mean count under p, averaged over the observations of fit."""
        _, _, mean, _ = self._moments(fit, p)
        w = np.asarray(fit.args[-1])[:fit.m]
        return float((w * np.log2(mean)).sum() / fit.n)


def build_models(fits: dict, bad: float, param_mode: str) -> dict:
    """A GroupModel for each allele at a BAD."""
    return {allele: GroupModel(fits[allele][bad], param_mode) for allele in ('ref', 'alt')}


def get_snvs_for_group(snvs: dict, group: set, min_samples: int = 0, min_cover: int = 0, max_cover: int = np.inf):
    """
    SNVs with at least min_samples samples from group.

    Samples of coverage max_cover or more are dropped. An SNV is kept if one
    of the remaining samples has a coverage of at least min_cover.
    """
    res = dict()
    for k, lt in snvs.items():
        t = [x for x in lt[1:] if x[0] in group and x[1] + x[2] < max_cover]
        if not t:
            continue
        if len(t) >= min_samples and max(v[1] + v[2] for v in t) >= min_cover:
            res[k] = [lt[0]] + t
    return res


def build_count_table(snvs: dict):
    count = defaultdict(lambda: defaultdict(int))
    for lt in snvs.values():
        lt = lt[1:]
        for t in lt:
            count[t[-1]][tuple(t[1:-1])] += 1
    return count


def count_dict_to_numpy(counts: dict):
    res = dict()
    for bad, counts in counts.items():
        r = list()
        for t in sorted(counts):
            r.append(list(t) + [counts[t]])
        res[bad] = np.array(r, dtype=int).reshape(-1, 3)
    return res


def build_count_tables(*snvs):
    """Tables of (ref, alt, n) rows by BAD for each group of SNVs, followed by those of all groups pooled."""
    counts = list(map(build_count_table, snvs))
    bads = set()
    for t in counts:
        bads |= set(t)
    base = dict()
    for bad in bads:
        c = defaultdict(int)
        for count in counts:
            for t, n in count.get(bad, dict()).items():
                c[t] += n
        base[bad] = c
    counts.append(base)
    return list(map(count_dict_to_numpy, counts))


_empty = np.zeros((0, 3), dtype=int)


def fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values. A nan stays nan and does not take part in the adjustment."""
    pvals = pd.to_numeric(pd.Series(pvals), errors='coerce').to_numpy(dtype=float)
    res = np.full(len(pvals), np.nan)
    inds = ~np.isnan(pvals)
    if inds.any():
        res[inds] = multitest.multipletests(pvals[inds], method='fdr_bh')[1]
    return res


_MODELS = dict()


def _set_models(models: dict):
    _MODELS.update(models)


def _run_task(fun, task: tuple):
    bad, t = task
    return fun(t, models=_MODELS[bad])


def run_tests(fun, tasks: list, models: dict, n_jobs: int) -> list:
    """
    Maps fun over the (BAD, task) pairs of tasks as fun(task, models=models[BAD]).

    One pool serves all of the BADs, and its workers receive the models once, at their start.
    """
    if n_jobs <= 1 or len(tasks) < 2:
        return [fun(t, models=models[bad]) for bad, t in tasks]
    n_jobs = min(n_jobs, len(tasks))
    with process_pool(n_jobs, initializer=_set_models, initargs=(models,)) as p:
        return p.map(partial(_run_task, fun), tasks, chunksize=max(1, len(tasks) // (4 * n_jobs)))


def _fitted_models(fits: dict, tasks_by_bad: dict, param_mode: str) -> dict:
    """GroupModels of the BADs with tasks, with a warning about those the fit lacks."""
    missing = sorted(set(tasks_by_bad) - set(fits['ref']))
    if missing:
        n = sum(len(tasks_by_bad[bad]) for bad in missing)
        logging.warning(f'The fit has no BAD {", ".join(map(str, missing))}: {n} tests there are not run.')
    return {bad: build_models(fits, bad, param_mode) for bad in sorted(tasks_by_bad) if bad in fits['ref']}


def _seed(snv) -> int:
    """A seed of the bootstrap that depends on the SNV only, so the result does not depend on n_jobs."""
    return zlib.crc32(repr(snv).encode())


def lrt_test(counts: tuple, models: dict, skip_failures=False, n_bootstrap=0, calc_es=False, es_mode='entropy',
             calc_pairs=False):
    """
    Likelihood ratio test of p being the same in all groups, for each allele.

    Parameters
    ----------
    counts : tuple
        An SNV identifier, a table of (ref, alt, n) rows for each group and
        the table of all groups pooled.
    models : dict
        A GroupModel for 'ref' and for 'alt'.
    skip_failures : bool, optional
        Return (None, identifier) if the test fails for both alleles. The default is False.
    n_bootstrap : int, optional
        Number of bootstrap samples for the bias correction. The default is 0.
    calc_es : bool, optional
        Compute an effect size and its variance for each group. The default is False.
    es_mode : str, optional
        'entropy' compares the means of log2 of a count under the p of a
        group and under the pooled p, 'mean' compares log2 of the mean
        counts. The default is 'entropy'.
    calc_pairs : bool, optional
        Test each group against the pooled p as well. The default is False.

    Returns
    -------
    res : list
        For each allele: the p-value, the p-values of the groups (with
        calc_pairs), the estimates of p of the groups, the pooled estimate,
        the effect sizes and their variances (with calc_es). All are nan for
        an allele where an estimate of p failed.
    info : list
        The identifier followed by the number of observations in each group.
    """
    snv = counts[0]
    counts = counts[1:]
    n_groups = len(counts) - 1
    rng = np.random.default_rng(_seed(snv))
    res = list()
    for allele in ('ref', 'alt'):
        model = models[allele]
        tables = counts if allele == 'ref' else [c[:, (1, 0, 2)] for c in counts]
        n_cols = 2 + n_groups * (1 + calc_pairs + 2 * calc_es)
        try:
            joint = model.fit(tables[-1], n_bootstrap=n_bootstrap, rng=rng)
            groups = [model.fit(t, n_bootstrap=n_bootstrap, rng=rng) for t in tables[:-1]]
        except (KeyError, IndexError):
            joint, groups = None, list()
        if joint is None or not (joint.success and all(g.success for g in groups)):
            res.append([float('nan')] * n_cols)
            continue
        stat = 2 * (joint.nll - sum(g.nll for g in groups))
        pvals = [chi2.sf(stat, n_groups - 1)] if n_groups > 1 else [float('nan')]
        es, variances = list(), list()
        for g in groups:
            if calc_pairs:
                pval = chi2.sf(2 * (model.nll(g.args, joint.p) - g.nll), 1)
                pvals.append(1.0 if np.isnan(pval) else pval)
            if calc_es:
                if es_mode == 'entropy':
                    mean, var = model.log_moments(g, g.p)
                    mean_all, var_all = model.log_moments(g, joint.p)
                    es.append(mean - mean_all)
                    variances.append((var + var_all) / g.n)
                elif es_mode == 'mean':
                    es.append(model.log_mean(g, g.p) - model.log_mean(g, joint.p))
                    variances.append(float('nan'))
                else:
                    es.append(float('nan'))
                    variances.append(float('nan'))
        res.append(pvals + [g.p for g in groups] + [joint.p] + es + variances)
    if skip_failures and all(np.isnan(r[0]) for r in res):
        return None, snv
    n = [c[:, -1].sum() for c in counts[:-1]]
    return res, [snv] + n


def transform_p(p, var):
    pt = np.log(p) - np.log1p(-p)
    var = var * (1 / p + 1 / (1 - p)) ** 2
    return pt, var


def wald_test(counts: Tuple[tuple, np.ndarray, np.ndarray, np.ndarray], models: dict, skip_failures=False,
              contrasts: Tuple[float, float, float] = (1, -1, 0), logit_transform=False, robust_se=False,
              n_bootstrap=0):
    """
    Wald test of a linear contrast of p of two groups, for each allele.

    Parameters
    ----------
    counts : tuple
        An SNV identifier, the tables of (ref, alt, n) rows of the two groups
        and the pooled table. The latter is not used.
    models : dict
        A GroupModel for 'ref' and for 'alt'.
    skip_failures : bool, optional
        The p-value of an allele is nan if an estimate or its variance fails,
        and (None, identifier) is returned if that happens for both alleles.
        Otherwise such a p-value is 1. The default is False.
    contrasts : tuple, optional
        (a, b, c) of the null hypothesis a * p_a + b * p_b + c = 0. The default is (1, -1, 0).
    logit_transform : bool, optional
        Test logits of p. A contrast with a single p, like (1, 0, -0.5), has
        its constant moved to the logit scale. In other contrasts the
        constant is taken as given. The default is False.
    robust_se : bool, optional
        Sandwich variance estimates. The default is False.
    n_bootstrap : int, optional
        Number of bootstrap samples for the bias correction. The default is 0.

    Returns
    -------
    res : list
        For each allele: p-value, both estimates of p and their standard errors.
    info : tuple
        The identifier and the numbers of observations in the two groups.
    """
    snv, counts_a, counts_b = counts[:3]
    rng = np.random.default_rng(_seed(snv))
    res = list()
    for allele in ('ref', 'alt'):
        model = models[allele]
        if allele == 'alt':
            counts_a = counts_a[:, (1, 0, 2)]
            counts_b = counts_b[:, (1, 0, 2)]
        try:
            fits = [model.fit(c, n_bootstrap=n_bootstrap, rng=rng) for c in (counts_a, counts_b)]
        except (KeyError, IndexError):
            res.append((float('nan'),) * 5)
            continue
        variances = list()
        for fit in fits:
            var = model.variance(fit, robust=robust_se) if np.isfinite(fit.p) else float('nan')
            # The standard error is not allowed to exceed the distance from p to 1.
            variances.append(min(var, (1.0 - fit.p) ** 2) if np.isfinite(var) else var)
        (a_p, b_p), (a_var, b_var) = (f.p for f in fits), variances
        correct = bool(np.isfinite(a_var) and np.isfinite(b_var) and a_var + b_var > 0)
        if not (correct and fits[0].success and fits[1].success):
            pval = float('nan') if skip_failures else 1.0
        else:
            a, b, eq = contrasts
            if logit_transform:
                a_p_t, a_var_t = transform_p(a_p, a_var)
                b_p_t, b_var_t = transform_p(b_p, b_var)
                for c in ((a, b), (b, a)):
                    if eq and c[0] and not c[1] and 0 < -eq / c[0] < 1:
                        eq = -c[0] * (np.log(-eq / c[0]) - np.log1p(eq / c[0]))
                        break
            else:
                a_p_t, a_var_t = a_p, a_var
                b_p_t, b_var_t = b_p, b_var
            stat = abs(a * a_p_t + b * b_p_t + eq)
            var = a ** 2 * a_var_t + b ** 2 * b_var_t
            pval = 2 * norm.sf(stat, scale=var ** 0.5, loc=0.0) if var > 0 else 1.0
        res.append((pval, a_p, b_p, a_var ** 0.5, b_var ** 0.5))
    if skip_failures and all(np.isnan(r[0]) for r in res):
        return None, snv
    n_a, n_b = counts_a[:, -1].sum(), counts_b[:, -1].sum()
    return res, (snv, n_a, n_b)


def _load_project(name: str, fit: str = None) -> tuple:
    """SNVs, scorefiles, fits and the compression of a project. fit is a path to a fit file of another project."""
    init_filename = get_init_file(name)
    compressor = init_filename.split('.')[-1]
    init = load_pickle(init_filename)
    fits = load_pickle(fit if fit else f'{name}.fit.{compressor}')
    return init['snvs'], init['scorefiles'], fits, compressor


def _save_result(name: str, kind: str, compressor: str, res: dict) -> dict:
    """Stores res in the file of the project for kind, next to the results kept there under other subnames."""
    filename = f'{name}.{kind}.{compressor}'
    if os.path.isfile(filename):
        res = {**load_pickle(filename), **res}
    save_pickle(res, filename)
    return res


def differential_test(name: str, group_a: List[str], group_b: List[str], mode='wald', min_samples=2, min_cover=0,
                      max_cover=np.inf, skip_failures=True, group_test=True, alpha=0.05, max_cover_group_test=None,
                      filter_chr=None, filter_id=None, contrasts=(1, -1, 0), subname=None, param_mode='window',
                      logit_transform=False, robust_se=False, n_bootstrap=0, fit: str = None, n_jobs=-1):
    """
    Tests each SNV for a difference in allele-specificity between two groups of samples.

    Parameters
    ----------
    name : str
        Project name.
    group_a, group_b : List[str]
        Filenames, folders or masks that select the samples of the control
        group and of the test group.
    mode : str, optional
        'wald' or 'lrt'. The default is 'wald'.
    min_samples : int, optional
        Minimal number of samples of a group at an SNV. The default is 2.
    min_cover : int, optional
        An SNV is tested if a sample of each group has at least this coverage. The default is 0.
    max_cover : int, optional
        Samples of this coverage or higher are dropped. The default is np.inf.
    skip_failures : bool, optional
        Leave out SNVs where the test fails for both alleles. An allele where
        it fails has nan p-values. The default is True.
    group_test : bool, optional
        Also test the groups as wholes, with all SNVs pooled, at each BAD. The default is True.
    alpha : float, optional
        FDR level. The Benjamini-Hochberg adjusted p-values do not depend on it. The default is 0.05.
    max_cover_group_test : int, optional
        Upper bound on the coverage in the test of the groups as wholes. The default is max_cover.
    filter_chr, filter_id : str, optional
        Regular expressions that the chromosome and the identifier of an SNV must match.
    contrasts : tuple, optional
        (a, b, c) of the Wald null hypothesis a * p_a + b * p_b + c = 0. The default is (1, -1, 0).
    subname : str, optional
        Key under which the result is stored in the project.
    param_mode : str, optional
        'window' takes r of each count from its window, 'line' from the
        window at the mean count. The default is 'window'.
    logit_transform : bool, optional
        Wald test on logits of p. The default is False.
    robust_se : bool, optional
        Sandwich variance in the Wald test. The default is False.
    n_bootstrap : int, optional
        Number of bootstrap samples for the bias correction of p. The default is 0.
    fit : str, optional
        Path to a fit file of another project.
    n_jobs : int, optional
        Number of parallel jobs, -1 for all cores. The default is -1.

    Returns
    -------
    dict
        Results of the project by subname. An entry has the table 'tests',
        the SNVs of both groups in 'snvs' and, with group_test, the table 'whole'.
    """
    max_cover = np.inf if max_cover is None else max_cover
    min_cover = 0 if min_cover is None else min_cover
    if max_cover_group_test is None:
        max_cover_group_test = max_cover
    if filter_chr is not None:
        filter_chr = re.compile(filter_chr)
    if filter_id is not None:
        filter_id = re.compile(filter_id)
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    snvs, scorefiles, fits, compressor = _load_project(name, fit)
    if type(group_a) is str:
        group_a = [group_a]
    if type(group_b) is str:
        group_b = [group_b]
    group_a = select_filenames(group_a, scorefiles)
    group_b = select_filenames(group_b, scorefiles)
    group_a = {scorefiles.index(f) for f in group_a}
    group_b = {scorefiles.index(f) for f in group_b}
    assert not (group_a & group_b), 'Groups should not intersect.'
    snvs_a = get_snvs_for_group(snvs, group_a, min_samples=min_samples, min_cover=min_cover, max_cover=max_cover)
    snvs_b = get_snvs_for_group(snvs, group_b, min_samples=min_samples, min_cover=min_cover, max_cover=max_cover)
    snvs = set(snvs_a) & set(snvs_b)
    if filter_id:
        snvs = {x for x in snvs if snvs_a[x][0][0] and filter_id.match(str(snvs_a[x][0][0]))}
    if filter_chr:
        snvs = {x for x in snvs if filter_chr.match(x[0])}
    snvs_a = {k: snvs_a[k] for k in snvs}
    snvs_b = {k: snvs_b[k] for k in snvs}
    if mode == 'lrt':
        cols = ['pval', 'p_control', 'p_test', 'p']
        test_fun = partial(lrt_test, n_bootstrap=n_bootstrap)
    else:
        cols = ['pval', 'p_control', 'p_test', 'std_control', 'std_test']
        test_fun = partial(wald_test, contrasts=contrasts, logit_transform=logit_transform, robust_se=robust_se,
                           n_bootstrap=n_bootstrap)
    cols = [f'{allele}_{c}' for allele in ('ref', 'alt') for c in cols]
    if group_test:
        whole_tables = build_count_tables(snvs_a, snvs_b)
    tasks_by_bad = defaultdict(list)
    for snv in sorted(snvs_a):
        tables = build_count_tables({snv: snvs_a[snv]}, {snv: snvs_b[snv]})
        for bad in tables[-1]:
            t = [c.get(bad, _empty) for c in tables]
            if len(t[0]) and len(t[1]):
                tasks_by_bad[bad].append((snv, *t))
    models = _fitted_models(fits, tasks_by_bad, param_mode)
    whole = list()
    for bad in models if group_test else ():
        tables = [c.get(bad, _empty) for c in whole_tables]
        tables = [c[c[:, 0] + c[:, 1] < max_cover_group_test] for c in tables]
        if len(tables[0]) and len(tables[1]):
            r, _ = test_fun(('all', *tables), models=models[bad], skip_failures=False)
            whole.append([bad] + list(r[0]) + list(r[1]))
    tasks = [(bad, t) for bad in models for t in tasks_by_bad[bad]]
    res = list()
    for (bad, _), (r, t) in zip(tasks, run_tests(partial(test_fun, skip_failures=skip_failures), tasks, models, n_jobs)):
        if r is not None:
            res.append([*t] + [bad] + list(r[0]) + list(r[1]))
    df = pd.DataFrame(res, columns=['ind', 'n_control', 'n_test', 'bad'] + cols)
    df['ref_fdr_pval'] = fdr(df['ref_pval'])
    df['alt_fdr_pval'] = fdr(df['alt_pval'])
    res = {subname: {'tests': df, 'snvs': (snvs_a, snvs_b)}}
    if group_test:
        res[subname]['whole'] = pd.DataFrame(whole, columns=['bad'] + cols)
    return _save_result(name, 'difftest', compressor, res)


def anova_test(name: str, groups: List[str], alpha=0.05, subname=None, fit=None, min_groups=2,
               min_samples=4, min_cover=0, max_cover=np.inf, param_mode='window', skip_failures=True,
               es_mode='entropy', n_jobs=1):
    """
    Tests each SNV for a difference in allele-specificity among several groups of samples.

    Parameters
    ----------
    name : str
        Project name.
    groups : List[str]
        A filename, folder or mask for each group. If empty, the groups are
        those that were combined under subnames in the project.
    alpha : float, optional
        FDR level. The Benjamini-Hochberg adjusted p-values do not depend on it. The default is 0.05.
    subname : str, optional
        Key under which the result is stored in the project.
    fit : str, optional
        Path to a fit file of another project.
    min_groups : int, optional
        Minimal number of groups with an SNV, at least 2 are required at a BAD. The default is 2.
    min_samples : int, optional
        Minimal number of samples of a group at an SNV. The default is 4.
    min_cover, max_cover : int, optional
        Bounds on the coverage, as in differential_test.
    param_mode : str, optional
        'window' or 'line', as in differential_test. The default is 'window'.
    skip_failures : bool, optional
        Leave out SNVs where the test fails for both alleles, as in differential_test. The default is True.
    es_mode : str, optional
        'entropy' or 'mean', as in lrt_test. The default is 'entropy'.
    n_jobs : int, optional
        Number of parallel jobs, -1 for all cores. The default is 1.

    Returns
    -------
    dict
        Results of the project by subname. An entry has the table 'tests'
        and the SNVs of each group in 'snvs'.
    """
    max_cover = np.inf if max_cover is None else max_cover
    min_cover = 0 if min_cover is None else min_cover
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    snvs, scorefiles, fits, compressor = _load_project(name, fit)
    if groups:
        groups = [[scorefiles.index(f) for f in select_filenames([pattern], scorefiles)] for pattern in groups]
        names = list(range(1, len(groups) + 1))
    else:
        combine_filename = f'{name}.comb.{compressor}'
        if not os.path.isfile(combine_filename):
            raise FileNotFoundError('No groups are given and the project has no combined p-values to take them from.')
        combine = load_pickle(combine_filename)
        groups, names = list(), list()
        for key, its in combine.items():
            if key:
                groups.append(list(its['groups']))
                names.append(key)
    if len(groups) < 2:
        raise ValueError('At least 2 groups are required: either pass them or run combine with a subname for each group.')
    if 'all' in map(str, names):
        raise ValueError("A group named 'all' clashes with the columns of all groups pooled: pass the groups explicitly.")
    shared = {f for i, a in enumerate(groups) for b in groups[i + 1:] for f in set(a) & set(b)}
    if shared:
        raise ValueError('Groups should not intersect, they share: ' + ', '.join(scorefiles[i] for i in sorted(shared)))
    snvs_groups = [get_snvs_for_group(snvs, set(g), min_samples=min_samples, min_cover=min_cover, max_cover=max_cover)
                   for g in groups]
    snvs_groups_count = defaultdict(int)
    for g in snvs_groups:
        for snv in g:
            snvs_groups_count[snv] += 1
    snvs = {snv for snv, n in snvs_groups_count.items() if n >= min_groups}
    snvs_groups = [{k: g.get(k, list()) for k in snvs} for g in snvs_groups]
    n_groups = len(snvs_groups)
    tasks_by_bad = defaultdict(list)
    group_inds = dict()
    for snv in sorted(snvs):
        tables = build_count_tables(*[{snv: g[snv]} for g in snvs_groups])
        for bad, pooled in tables[-1].items():
            inds = [i for i in range(n_groups) if len(tables[i].get(bad, _empty))]
            # A test at a BAD needs 2 groups, whatever min_groups is.
            if len(inds) >= max(2, min_groups):
                tasks_by_bad[bad].append(((snv, bad), *[tables[i][bad] for i in inds], pooled))
                group_inds[(snv, bad)] = np.array(inds)
    models = _fitted_models(fits, tasks_by_bad, param_mode)
    tasks = [(bad, t) for bad in models for t in tasks_by_bad[bad]]
    test_fun = partial(lrt_test, skip_failures=skip_failures, calc_es=True, es_mode=es_mode, calc_pairs=True)
    res = list()
    for (bad, _), (r, t) in zip(tasks, run_tests(test_fun, tasks, models, n_jobs)):
        if r is None:
            continue
        inds = group_inds[t[0]]
        m = len(inds)
        row = list()
        for rt in r:
            # The layout of rt is that of the result of lrt_test.
            st = np.full(4 * n_groups + 2, None, dtype=object)
            st[-1] = rt[0]
            st[inds] = rt[1: 1 + m]
            st[inds + n_groups] = rt[1 + m: 1 + 2 * m]
            st[2 * n_groups] = rt[1 + 2 * m]
            st[2 * n_groups + 1 + inds] = rt[2 + 2 * m: 2 + 3 * m]
            st[3 * n_groups + 1 + inds] = rt[2 + 3 * m: 2 + 4 * m]
            row += list(st)
        n = np.zeros(n_groups + 1, dtype=int)
        n[inds] = t[1:]
        n[-1] = sum(t[1:])
        res.append([t[0][0]] + list(n) + [bad] + row)
    cols = [f'pval_{i}' for i in names] + [f'p_{i}' for i in names] + ['p_all'] + [f'es_{i}' for i in names] \
        + [f'es_var_{i}' for i in names] + ['pval']
    cols = ['ind'] + [f'n_{i}' for i in names] + ['n_all', 'bad'] + [f'ref_{c}' for c in cols] + [f'alt_{c}' for c in cols]
    df = pd.DataFrame(res, columns=cols)
    for allele in ('ref', 'alt'):
        for n in names:
            df[f'{allele}_fdr_pval_{n}'] = fdr(df[f'{allele}_pval_{n}'])
    for allele in ('ref', 'alt'):
        df[f'{allele}_fdr_pval'] = fdr(df[f'{allele}_pval'])
    res = {subname: {'tests': df, 'snvs': snvs_groups}}
    return _save_result(name, 'anova', compressor, res)
