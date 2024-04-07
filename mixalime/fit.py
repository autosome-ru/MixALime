# -*- coding: utf-8 -*-
import multiprocessing as mp
import numpy as np
import logging
import dill
import os
from betanegbinfit.utils import collect_stats
from collections import defaultdict
from itertools import product
from functools import partial
from .utils import openers, get_init_file, get_model_creator


def _finalize_fit(model, fit):
    params_to_save = ('mu', 'b', 'mu_k', 'w', 'p1', 'p2')
    std = fit.get('std', None)
    for p in params_to_save:
        for s in model.slices:
            pn = f'{p}{s}'
            if pn not in fit:
                try:
                    fit[pn] = model.get_param(p, model.prev_res)
                except KeyError:
                    continue
                if std is not None:
                    std[pn] = 0.0


def _run(aux: tuple, data: dict, left: int,
         max_count: int, mod: str, dist: str, 
         estimate_p: bool, window_size: int, 
         apply_weights: bool, window_behavior: str, min_slices: int,
         start_est=True, compute_pdf=False, k_left_bound=1,
         adjusted_loglik=False, adjust_line=False, regul_alpha=0.0, 
         regul_n=True, regul_slice=True, regul_prior='laplace', std=False, 
         fix_params=str(), optimizer='SLSQP', r_transform=None,
         symmetrify=False, small_dataset_strategy='conservative',
         small_dataset_n=1000, stop_slice_n=10, kappa_right=None, use_cpu=False):
    if use_cpu:
        os.environ["JAX_PLATFORM_NAME"] = 'cpu'
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = 'cpu'
    bad, switch = aux
    data = data[bad]
    if data[:, -1].sum() < small_dataset_n:
        if small_dataset_strategy == 'conservative':
            if dist == 'BetaNB':
                dist = 'NB'
            fix_params = 'b=1;mu=0;w=1;' + fix_params
        elif small_dataset_strategy == 'fix_r':
            fix_params = 'b=1;mu=0;' + fix_params
    prefix = 'alt_' if switch else 'ref_'
    logging.info(f'[BAD={bad}, {prefix[:-1]}] Optimization...')
    if switch:
        data = np.array(data, dtype=float)[:, [1, 0, 2]]
    inst_params = {'name': mod, 'bad': bad, 'left': left, 'dist': dist, 'estimate_p': estimate_p,
                   'left_k': k_left_bound, 'start_est': start_est, 'apply_weights': apply_weights,
                   'window_size': window_size, 'left_k': k_left_bound,
                   'window_behavior': window_behavior, 'min_slices': min_slices,
                   'adjust_line': adjust_line, 'start_est': start_est,
                   'apply_weights': apply_weights, 'regul_alpha': regul_alpha,
                   'regul_n': regul_n, 'regul_slice': regul_slice,
                   'regul_prior': regul_prior, 'fix_params': fix_params,
                   'r_transform': r_transform, 'symmetrify': symmetrify,
                   'kappa_right': kappa_right}
    model = get_model_creator(**inst_params)()
    fit = model.fit(data, calc_std=std, optimizer=optimizer, stop_slice_n=stop_slice_n)
    _finalize_fit(model, fit)
    stds = list()
    ests = list()
    names = list()
    if 'std' in fit:
        for n, std in fit['std'].items():
            fn = fit[n]
            if np.isscalar(fn):
                ests.append(fn)
                stds.append(std)
                names.append(n)
            else:
                for i, s in enumerate(model.slices):
                    ests.append(fn[i])
                    stds.append(std[i])
                    names.append(f'{n}{s}')
        params = {'names': names, 'ests': ests, 'stds': stds}
    else:
        for n, est in fit.items():
            if np.isscalar(est):
                ests.append(est)
                names.append(n)
            else:
                for i, s in enumerate(model.slices):
                    ests.append(est[i])
                    names.append(f'{n}{s}')
        params = {'names': names, 'ests': ests}
    logging.info(f'[BAD={bad}, {prefix[:-1]}] Calculating fit indices...')
    # data = data if not symmetrify else odata
    stats = collect_stats(model, calc_pvals=False, calc_es=False, calc_adjusted_loglik=adjusted_loglik)
    r =  {'params': params, 'stats': stats, 'inst_params': inst_params}
    return r
    

def fit(name: str, model='line', dist='BetaNB', left=None,
        estimate_p=False, window_size=1000,
        window_behavior='both', min_slices=1, adjust_line=False, k_left_bound=1,
        max_count=np.inf, max_cover=np.inf, apply_weights=False,  start_est=True,
        adjusted_loglik=False, regul_alpha=0.0, regul_n=True, regul_slice=True, 
        regul_prior='laplace', std=False, fix_params=str(), optimizer='SLSQP', 
        r_transform=None, symmetrify=False, small_dataset_strategy='conservative',
        small_dataset_n=1000, stop_slice_n=10, kappa_right=None, n_jobs=1):
    """

    Parameters
    ----------
    data : pd.DataFrame
        Pandas DataFrame.
    output_folder : str
        Name of output folder where results will be stored.
    bads : list, optional
        List of BADs. If None, then bads will be guessed from the table. The
        default is None.
    model : str, optional
        Model name. Currently, can be either 'line' (ModelLine) or 'window'
        (ModelWindow). The default is 'line'.
    dist : str, optional
        Which mixture distribution to use. Can be either 'NB' or 'BetaNB'. The
        default is 'BetaNB'.
    left : int, optional
        Left-truncation bound. If None, then it will be estimated from the counts data as the minimal present count - 1. The default is None.
    estimate_p : bool, optional
        If True, then p will be estimated instead of assuming it to be fixed to
        bad / (bad + 1). The default is False.
    window_size : int, optional
        Has effect only if model = 'window', sets the required window size. The
        default is 1000.
    max_count : int, optional
        Maximal number of counts for an allele (be it ref or alt). The default is np.inf.
    max_cover : int, optional
        Maximal sum of ref + alt. The default is np.inf
    adjusted_loglik: bool, optional
        If True, then adjusted loglikelihood is also calculated alongside other statistics. The default is False.
    regul_alpha: float, optional
        Alpha multiplier of regularization/prior part of the objective
        function for the concentration parameter kappa. Vaid only for model='window'. The default is 0.0.
    regul_n: bool, optional
        If True, then alpha is multiplied by a number of observations captured
        by a particular window. Vaid only for model='window'. The default is True.
    regul_slice: bool, optional
       If True, then alpha is multiplied by an average slice number captured
       by a particular window. Vaid only for model='window'. The default is True.
   regul_prior: bool, optional
       A name of prior distribution used to penalize small kappa values. Can
       be 'laplace' (l1) or 'normal' (l2). Vaid only for model='window'. The default is 'laplace'.
   kappa_right: float, optional
       Right boundary for the kappa parameter. The default is None.
    n_jobs : int, optional
        Number of parallel jobs to run. If -1, then it is determined
        automatically. The default is -1.

    Returns
    -------
    None

    """
    init_filename = get_init_file(name)
    compression = init_filename.split('.')[-1]
    open = openers[compression]
    with open(init_filename, 'rb') as f:
        data = dill.load(f)['counts']
    if left is None:
        left = -1
    data = {bad : mx[(mx[:, 0] < max_count) & (mx[:, 1] < max_count) & \
                      (mx[:, 0] > left) & (mx[:, 1] > left) & ((mx[:, 0] + mx[:, 1]) < max_cover), :] 
            for bad, mx in data.items()}
    
    for bad, mx in data.items():
        n = mx[:, -1].sum()
        if n < window_size:
            s = f'Total number of samples at BAD {bad} is less than a window size ({n} < {window_size}).'
            if n < small_dataset_n:
                s += f' Number of samples is too small for a sensible fit, a {small_dataset_strategy} fit will be used.'
            logging.warning(s)
    if left == -1:
        left = min(data[bad][:, [0, 1]].min() for bad in data) - 1
    aux = list(product(sorted(data), (False, True)))
    if n_jobs == -1:
        n_jobs = max(1, mp.cpu_count())
    n_jobs = min(len(aux), n_jobs)
    fun = partial(_run, data=data, left=left, 
                  mod=model, dist=dist, 
                  window_behavior=window_behavior, min_slices=min_slices,
                  apply_weights=apply_weights,
                  start_est=start_est, max_count=max_count,
                  k_left_bound=k_left_bound,
                  window_size=window_size, estimate_p=estimate_p,
                  adjusted_loglik=adjusted_loglik,
                  adjust_line=adjust_line,
                  regul_alpha=regul_alpha,
                  regul_n=regul_n,
                  regul_slice=regul_slice,
                  regul_prior=regul_prior,
                  std=std,
                  fix_params=fix_params,
                  optimizer=optimizer,
                  r_transform=r_transform,
                  symmetrify=symmetrify,
                  small_dataset_strategy=small_dataset_strategy,
                  small_dataset_n=small_dataset_n,
                  stop_slice_n=stop_slice_n,
                  kappa_right=kappa_right,
                  use_cpu=(n_jobs != 1))
    result = defaultdict(lambda: defaultdict())
    ralt = {True: 'alt', False: 'ref'}
    if n_jobs == 1:
        for (bad, alt), res in zip(aux, map(fun, aux)):
            result[ralt[alt]][bad] = res
    else:
        ctx = mp.get_context("forkserver")
        with ctx.Pool(n_jobs) as p:
            for (bad, alt), res in zip(aux, p.map(fun, aux)):
                result[ralt[alt]][bad] = res
    fit_filename = f'{name}.fit.{compression}'
    for f in ('test', 'comb', 'difftest'):
        filename = f'{name}.{f}.{compression}'
        if os.path.isfile(filename):
            os.remove(filename)
    with open(fit_filename, 'wb') as f:
        dill.dump(result, f)
