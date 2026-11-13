# -*- coding: utf-8 -*-
import multiprocessing as mp
import numpy as np
import logging
import os
from collections import defaultdict
from itertools import product
from functools import partial
from .utils import get_init_file, dictify_fix, load_pickle, save_pickle, process_pool
from .stats import fit_indices
from .models import ModelWindow


def _run(aux: tuple, data: dict, left: int, dist: str, estimate_p: bool, window_size: int,
         window_behavior: str, min_slices: int, k_left_bound=1, adjusted_loglik=False, adjust_line=False,
         regul_alpha=0.0, regul_n=True, regul_slice=True, regul_prior='laplace', std=False,
         fix_params=str(), optimizer='SLSQP', r_transform=None, symmetrify=False,
         small_dataset_strategy='conservative', small_dataset_n=1000, stop_slice_n=10, kappa_right=None,
         window_ratio=None):
    """
    Fits the model of one allele at one BAD. aux is (bad, switch); with switch the alt allele is modelled.

    The stats of a slice get window_lo and window_hi, the first and the last
    slice of the window whose fit gives the estimates there.
    """
    bad, switch = aux
    data = data[bad]
    if data[:, -1].sum() < small_dataset_n:
        if small_dataset_strategy == 'conservative':
            if dist == 'BetaNB':
                dist = 'NB'
            fix_params = 'b=1;mu=0;w=1;' + fix_params
        elif small_dataset_strategy in ('fixed_r', 'fix_r'):
            fix_params = 'b=1;mu=0;' + fix_params
    allele = 'alt' if switch else 'ref'
    logging.info(f'[BAD={bad}, {allele}] Optimization...')
    if switch:
        data = data[:, [1, 0, 2]]
    inst_params = {'name': 'window', 'bad': bad, 'left': int(left), 'dist': dist, 'estimate_p': estimate_p,
                   'left_k': k_left_bound, 'window_size': window_size, 'window_behavior': window_behavior,
                   'min_slices': min_slices, 'adjust_line': adjust_line, 'regul_alpha': regul_alpha,
                   'regul_n': regul_n, 'regul_slice': regul_slice, 'regul_prior': regul_prior,
                   'fix_params': fix_params, 'r_transform': r_transform, 'symmetrify': symmetrify,
                   'kappa_right': kappa_right, 'window_ratio': window_ratio}
    kwargs = {k: v for k, v in inst_params.items() if k != 'name'}
    kwargs['fix_params'] = dictify_fix(fix_params)
    model = ModelWindow(**kwargs)
    fit = model.fit(data, calc_std=std, optimizer=optimizer, stop_slice_n=stop_slice_n)
    stds = fit.pop('std', None)
    params = {'names': list(fit), 'ests': list(fit.values())}
    if stds is not None:
        params['stds'] = [stds[n] for n in fit]
    logging.info(f'[BAD={bad}, {allele}] Calculating fit indices...')
    stats = fit_indices(model.data, params, bad, int(left), dist, r_transform=r_transform, estimate_p=estimate_p,
                        adjusted_loglik=adjusted_loglik)
    for s, (a, b) in zip(model.slices, model.sources):
        stats[int(s)]['window_lo'], stats[int(s)]['window_hi'] = int(model.slices[a]), int(model.slices[b])
    return {'params': params, 'stats': stats, 'inst_params': inst_params}


def fit(name: str, dist='BetaNB', left=None, estimate_p=False, window_size=1000,
        window_behavior='both', min_slices=1, adjust_line=False, k_left_bound=1,
        max_count=np.inf, max_cover=np.inf, adjusted_loglik=False, regul_alpha=0.0, regul_n=True,
        regul_slice=True, regul_prior='laplace', std=False, fix_params=str(), optimizer='SLSQP',
        r_transform=None, symmetrify=False, small_dataset_strategy='conservative',
        small_dataset_n=1000, stop_slice_n=10, kappa_right=None, window_ratio=None, n_jobs=1):
    """
    Fits the window model to the counts of a project, for each BAD and each allele.

    Parameters
    ----------
    name : str
        Project name.
    dist : str, optional
        Distribution of the mixture components: 'NB', 'BetaNB' or 'MCNB'.
        The default is 'BetaNB'.
    left : int, optional
        Left-truncation bound. If None, it is the minimal count present in
        the data minus 1. The default is None.
    estimate_p : bool, optional
        If True, p is estimated, otherwise it is fixed to bad / (bad + 1).
        The default is False.
    window_size : int, optional
        Minimal number of observations in a window. The default is 1000.
    window_behavior : str, optional
        'both' grows a window in both directions, 'right' to the right first.
        The default is 'both'.
    min_slices : int, optional
        Minimal number of slices in a window. The default is 1.
    adjust_line : bool, optional
        Pull b and mu of each window towards those of the previous window.
        The default is False.
    k_left_bound : float, optional
        Lower bound of the concentration parameter kappa, raised to BAD + 1.001
        for BetaNB with r_transform and a fixed p. The default is 1.
    max_count : int, optional
        Counts of an allele are below this. The default is np.inf.
    max_cover : int, optional
        Sums of ref and alt are below this. The default is np.inf.
    adjusted_loglik : bool, optional
        Also compute the adjusted log-likelihood. The default is False.
    regul_alpha : float, optional
        Strength of the prior that favours high kappa. The default is 0.0.
    regul_n : bool, optional
        Multiply alpha by the number of observations in a window. The default is True.
    regul_slice : bool, optional
        Multiply alpha by the slice at the centre of a window. The default is True.
    regul_prior : str, optional
        'laplace' or 'normal'. The default is 'laplace'.
    std : bool, optional
        Compute standard errors of the estimates. The default is False.
    fix_params : str, optional
        Parameters fixed to constants, as 'name1=value1;name2=value2'.
    optimizer : str, optional
        'SLSQP' or 'L-BFGS-B'. The default is 'SLSQP'.
    r_transform : str, optional
        'NB', 'mean' or None. The default is None.
    symmetrify : bool, optional
        Symmetrify the counts before the fit. The default is False.
    small_dataset_strategy : str, optional
        What to do with a BAD of fewer than small_dataset_n observations:
        'conservative' fixes b=1, mu=0, w=1 and replaces BetaNB by NB,
        'fixed_r' fixes b=1, mu=0, 'none' does nothing. The default is 'conservative'.
    small_dataset_n : int, optional
        The default is 1000.
    stop_slice_n : int, optional
        A slice with fewer observations takes the estimates of the last fitted
        window. The default is 10.
    kappa_right : float, optional
        Upper bound of kappa. The default is None.
    window_ratio : float, optional
        The window of a slice s holds no slice below s / window_ratio or above
        s * window_ratio, and so fewer than window_size observations where the
        slices near s are sparse. The default is None, for no such bound.
    n_jobs : int, optional
        Number of parallel jobs, -1 for all cores. The default is 1.

    Returns
    -------
    dict
        The fit, also saved to the project.
    """
    if n_jobs != -1 and not (isinstance(n_jobs, (int, np.integer)) and n_jobs >= 1):
        raise ValueError(f'n_jobs should be -1 or a positive integer, got {n_jobs}.')
    init_filename = get_init_file(name)
    compression = init_filename.split('.')[-1]
    data = load_pickle(init_filename)['counts']
    lower = -1 if left is None else left
    data = {bad: mx[(mx[:, 0] < max_count) & (mx[:, 1] < max_count) & (mx[:, 0] > lower) & (mx[:, 1] > lower)
                    & ((mx[:, 0] + mx[:, 1]) < max_cover), :]
            for bad, mx in data.items()}
    for bad in [bad for bad, mx in data.items() if not len(mx)]:
        logging.warning(f'No observations at BAD {bad} pass the filters, the BAD is skipped.')
        del data[bad]
    if not data:
        raise ValueError('No observations pass the filters.')
    for bad, mx in data.items():
        n = mx[:, -1].sum()
        if n < window_size:
            s = f'Total number of samples at BAD {bad} is less than a window size ({n} < {window_size}).'
            if n < small_dataset_n:
                s += f' Number of samples is too small for a sensible fit, a {small_dataset_strategy} fit will be used.'
            logging.warning(s)
    if left is None:
        left = min(mx[:, [0, 1]].min() for mx in data.values()) - 1
    aux = list(product(sorted(data), (False, True)))
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count())
    n_jobs = min(len(aux), n_jobs)
    fun = partial(_run, data=data, left=left, dist=dist, window_behavior=window_behavior, min_slices=min_slices,
                  k_left_bound=k_left_bound, window_size=window_size, estimate_p=estimate_p,
                  adjusted_loglik=adjusted_loglik, adjust_line=adjust_line, regul_alpha=regul_alpha,
                  regul_n=regul_n, regul_slice=regul_slice, regul_prior=regul_prior, std=std,
                  fix_params=fix_params, optimizer=optimizer, r_transform=r_transform, symmetrify=symmetrify,
                  small_dataset_strategy=small_dataset_strategy, small_dataset_n=small_dataset_n,
                  stop_slice_n=stop_slice_n, kappa_right=kappa_right, window_ratio=window_ratio)
    result = defaultdict(dict)
    if n_jobs == 1:
        it = map(fun, aux)
    else:
        with process_pool(n_jobs) as p:
            it = p.map(fun, aux)
    for (bad, alt), res in zip(aux, it):
        result['alt' if alt else 'ref'][bad] = res
    result = dict(result)
    result['model'] = (dist, None if dist != 'BetaNB' or regul_alpha == 0 else regul_alpha)
    for f in ('test', 'comb', 'difftest', 'anova'):
        filename = f'{name}.{f}.{compression}'
        if os.path.isfile(filename):
            os.remove(filename)
    save_pickle(result, f'{name}.fit.{compression}')
    return result
