#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .utils import get_init_file, dictify_params, get_model_creator, openers
from betanegbinfit.distributions import LeftTruncatedBinom, LeftTruncatedBetaBinom
from betanegbinfit.utils import get_params_at_slice
from multiprocessing import cpu_count, Pool
from scipy.optimize import minimize_scalar
from collections import defaultdict
from functools import partial
from itertools import product
from gmpy2 import mpfr
import numpy as np
import dill
import os


def calc_stats(t: tuple, inst_params: dict, params: dict, swap: bool,
               correction: str = None, gof_tr=None, dataset_n_thr=float('inf'),
               max_size=None):
    alt, counts, rmsea, dataset_n = t
    res = list()
    params = get_params_at_slice(params, alt, clip_at_max_slice=False)
    bad = inst_params['bad']
    if (gof_tr is not None and rmsea.get(alt, np.inf) > gof_tr) or (dataset_n.get(alt, 0) < dataset_n_thr):
        params['r'] = alt
        if 'k' in params:
            inst_params = inst_params.copy()
            inst_params['dist'] = 'NB'
        if 'w' in params:
            params['w'] = 1.0
        if 'p1' in params:
            params['p1'] = bad / (bad + 1)
        if 'p2' in params:
            params['p2'] = 1 / (bad + 1)
            
    if not hasattr(calc_stats, '_cache'):
        calc_stats._cache = dict()
    cache = calc_stats._cache
    key = tuple(inst_params.values())
    if key not in cache:
        cache[key] = get_model_creator(**inst_params)()
    model = cache[key]
    w = params.get('w', 0)
    params = model.dict_to_vec(params)
    iter_w = False
    if w and w != 1:
        if correction and max_size:
            m = len(counts)
            counts = np.pad(counts, (0, max_size - m))
        if correction == 'hard':
            lpdf, rpdf = model.logprob_modes(params, counts)
            lpdf = lpdf[:m]; rpdf = rpdf[:m]; counts = counts[:m]
            w = np.array(lpdf > rpdf, dtype=int)
            iter_w = True
        elif correction == 'single':
            lpdf, rpdf = model.logprob_modes(params, counts)
            lpdf = lpdf[:m]; rpdf = rpdf[:m]; counts = counts[:m]
            odds_old = w / (1 - w)
            odds_new = odds_old * np.exp(lpdf - rpdf)
            w = odds_new / (odds_new + 1)
            w = np.array(w, dtype=float)
            w[np.isinf(odds_new)] = 1.0
            iter_w = True
    cdfs = model.cdf_modes(params, counts - 1)
    if not iter_w:
        mean = model.mean(params)
    else:
        mean_l, mean_r = model.mean(params, return_modes=True)
    for it in zip(*cdfs, counts, w) if iter_w else zip(*cdfs, counts):
        if iter_w:
            w = it[-1]
            mean = w * mean_l + (1 - w) * mean_r
        else:
            w = mpfr(w)
        cdf_l, cdf_r, c = it[:3]
        cdf = w * cdf_l + (1 - w) * cdf_r
        
        res.append((float(1 - cdf), np.log2(c) - np.log2(mean), ))
    if swap:
        res = {(alt, c): v for c, v in zip(counts, res)}
    else:
        res = {(c, alt): v for c, v in zip(counts, res)}
    return res

def calc_stats_binom(t: tuple, w: str, bad: float, left: int, swap: bool, params=None):
    alt, counts = t
    n = counts + alt
    res = list()
    if params:
        p = params[bad]['alt' if swap else 'ref']
    else:
        p = None
    dist = LeftTruncatedBinom
    if w is None or bad == 1:
        p = p if p else 0.5
        pv = np.array(list(map(float, dist.long_sf(counts - 1, n, p, left))))
        es = np.log2(counts) - np.log2(list(map(float, dist.mean(n, p, left))))
        for pv, es in zip(pv, es):
            res.append((pv, es))
    else:
        tw = w
        if p is None:
            p = bad / (bad + 1)
        cdf1 = dist.long_cdf(counts - 1, n, p, left)
        cdf2 = dist.long_cdf(counts - 1, n, 1 - p, left)
        mean_l = dist.mean(n, p, left)
        mean_r = dist.mean(n, 1 - p, left)
        for cdf_l, cdf_r,  mean_r, mean_l, c, n in zip(cdf1, cdf2, mean_l, mean_r, counts, n):
            w = float(tw)
            mean = w * mean_l + (1 - w) * mean_r
            cdf = w * cdf_l + (1 - w) * cdf_r
            mean = w * mean_l + (1 - w) * mean_r
            res.append((float(1 - cdf), np.log2(c) - np.log2(mean), ))
            
    if swap:
        res = {(alt, c): v for c, v in zip(counts, res)}
    else:
        res = {(c, alt): v for c, v in zip(counts, res)}
    return res

def calc_stats_betabinom(t: tuple, w: str, bad: float, left: int, swap: bool, params=None):
    dist = LeftTruncatedBetaBinom
    alt, counts = t
    n = counts + alt
    k = params[bad]['alt' if swap else 'ref']
    if np.iterable(k):
        p, k = k
    else:
        p = None
 
    def sf(p):
        res = np.empty(len(n), dtype=object)
        for nv in np.unique(n):
            ind = n == nv
            res[ind] = dist.long_sf(counts[ind] - 1, nv, p, k, left)
        return res
            
    res = list()
    if w is None or bad == 1:
        if p is None:
            p = 0.5
        pv = np.array(list(map(float, sf(p))))
        es = np.log2(counts) - np.log2(list(map(float, dist.mean(n, p, k, left))))
        for pv, es in zip(pv, es):
            res.append((pv, es))
    else:
        if p is None:
            p = bad / (bad + 1)
        tw = w
        cdf1 = sf(p)
        cdf2 = sf(1 - p)
        mean_l = dist.mean(n, p, k, left)
        mean_r = dist.mean(n, 1 - p, k, left)
        for cdf_l, cdf_r,  mean_r, mean_l, c, n in zip(cdf1, cdf2, mean_l, mean_r, counts, n):
            w = float(tw)
            mean = w * mean_l + (1 - w) * mean_r
            cdf = w * cdf_l + (1 - w) * cdf_r
            mean = w * mean_l + (1 - w) * mean_r
            res.append((float(1 - cdf), np.log2(c) - np.log2(mean), ))
            
    if swap:
        res = {(alt, c): v for c, v in zip(counts, res)}
    else:
        res = {(c, alt): v for c, v in zip(counts, res)}
    return res

def log_q(a, b, q=1):
    if q == 1:
        return a - b
    # x = np.exp(a) / np.exp(b)
    def q_fun(x):
        return (x ** (1 - q) - 1) / (1 - q)
    a = np.exp(a)
    b = np.exp(b)
    a, b =  q_fun(a), q_fun(1/b) 
    return a + b + (1 - q) * a * b
    # return  (x ** (1 - q) - 1) / (1 - q)

def est_binom_params(counts_d: dict, left: int, w: float, dist: str, est_p=False, max_cover:int=float('inf'),
                     inv_kl=False, n_jobs:int=1):
    if dist == 'binom':
        logprob = lambda x, r, p, k: LeftTruncatedBinom.logprob(x, r=r, p=p, left=left)
    else:
        logprob = lambda x, r, p, k: LeftTruncatedBetaBinom.logprob(x, r=r, mu=p, concentration=k, left=left)
    res = defaultdict(dict)
    def est(t, est_k=False):
        bad, swap, p = t
        counts = counts_d[bad]
        if swap:
            counts = counts[:, [1, 0, 2]]
        r = counts[:, 0] + counts[:, 1]
        ind = r <= max_cover
        r = r[ind]
        counts = counts[ind, :]
        
        uniq, inv = np.unique(r, return_inverse=True)
        uniq_x, inv_x = np.unique(counts[:, 0], return_inverse=True)
        pdf_emp = np.zeros_like(r, dtype=float)
        slice_mult = np.zeros_like(pdf_emp)
        n_total = counts[:, -1].sum()
        # slice_indices = list()
        counts[:, -1] = r
        for i, n in enumerate(uniq):
            ind = inv == i
            n = counts[ind, 2].sum()
            slice_mult[ind] = np.log(n) - np.log(n_total)
            pdf_emp[ind] = np.log(counts[ind, 2]) - np.log(n)
            # int_inds = np.where(ind)[0]
            # inds = int_inds[np.unique(counts[ind, 0], return_index=True)[1]]
            # z = np.zeros_like(ind, dtype=bool)
            # z[inds] = True
            # slice_indices.append((ind, z))
        # pdf_emp = np.log(pdf_emp)
        def fun(x, k=500, p=p):
            if est_k:
                k = x
            else:
                p = x
            if w is None:
                lp = logprob(counts[:, 0], r=r, p=p, k=k) 
            else:
                t1 = logprob(counts[:, 0], r=r, p=p, k=k) 
                t2 = logprob(counts[:, 0], r=r, p=1 - p, k=k)
                lp = np.log((1 - w) * np.exp(t1) + w * np.exp(t2))
            if inv_kl:
                q = 1
                lp = np.array(lp)
                # for ind, uq_x in slice_indices:
                #     lp[ind] -= np.log(np.clip(np.exp(lp[uq_x]).sum(), 1e-300, 1))
                lp = lp + slice_mult
                a = 0.5
                # loglik = lp * pdf_emp
                loglik = -np.exp(lp) * (-lp - pdf_emp)
                # loglik = - (1 - a)*np.exp(lp) * log_q(lp, pdf_emp, q=q) + a * lp * counts[:, -1] / counts[:, -1].sum()
            else:
                loglik = lp * counts[:, -1]
            return -np.sum(loglik)
        if est_k:
            a = 0.0
            b = 1000.0
            n = 50
        else:
            a = 0.01
            b = 0.99
            n = 10
        seps = np.linspace(a, b, n)[1:-1]
        best_f = float('inf')
        for i, k in enumerate(seps):
            f = fun(k)
            if f < best_f:
                best_f = f
                min_i = i
        seps = [a] + list(seps) + [b]
        a = seps[min_i]
        b = seps[min_i + 2]
        return minimize_scalar(fun, bounds=(a, b), method='bounded').x
    its = list(product(list(counts_d.keys()), (False, True)))
    its = [[bad, swap, bad / (bad + 1)] for bad, swap in its]
    
    # with Pool(n_jobs) as p:
    if est_p:
        for i, prob in enumerate(map(est, its)):
            bad, swap, _ = its[i]
            res[bad]['alt' if swap else 'ref'] = prob
            its[i][-1] = prob
    if dist != 'binom':
        est_k = partial(est, est_k=True)
        for i, k in enumerate(map(est_k, its)):
            bad, swap, _ = its[i]
            res[bad]['alt' if swap else 'ref'] = (its[i][-1], k) 
    return res


def test(name: str, correction: str = None, gof_tr: float = None, dataset_n_thr : int = float('inf'),
         fit: str = None, n_jobs: int = -1):
    n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    open = openers[compressor]
    if fit:
        comp_fit = fit.split('.')[-1]
        open2 = openers[comp_fit]
    else:
        fit = f'{name}.fit.{compressor}'
        open2 = open
    fit = fit if fit else f'{name}.fit.{compressor}'
    with open(filename, 'rb') as init,  open2(fit, 'rb') as fit:
        fit = dill.load(fit)
        counts_d = dill.load(init)['counts']
    del init
    res = dict()
    for bad in counts_d:
        for allele in ('ref', 'alt'):
            sub_res = dict()
            if allele not in res:
                res[allele] = dict()
            res[allele][bad] = sub_res
            swap = allele != 'ref'        
            inst_params = fit[allele][bad]['inst_params']
            inst_params['name'] = 'slice'
            params = dictify_params(fit[allele][bad]['params'])
            st = fit[allele][bad]['stats']
            counts = counts_d[bad][:, (1, 0) if swap else (0, 1)]
            alt = counts[:, 1]
            sub_c = [(u, counts[alt == u, 0], st.get(u, {'rsmea': np.nan}), st.get(u, {'n': 0})) for u in np.unique(alt)]
            max_size = max(map(lambda x: len(x[1]), sub_c))
            chunksize = int(np.ceil(len(sub_c) / n_jobs))
            with Pool(n_jobs) as p:
                f = partial(calc_stats, inst_params=inst_params, params=params, gof_tr=gof_tr, correction=correction, swap=swap,
                            max_size=max_size, dataset_n_thr=dataset_n_thr)
                it = p.imap_unordered(f, sub_c, chunksize=chunksize) if n_jobs > 1 else map(f, sub_c)
                for r in it:
                    sub_res.update(r)
    filename = f'{name}.comb.{compressor}'
    if os.path.isfile(filename):
        os.remove(filename)
    with open(f'{name}.test.{compressor}', 'wb') as f:
        dill.dump(res, f)
    return res


def binom_test(name: str, w: float, beta=False, estimate_p=False, max_cover=float('inf'), inv_kl=False, n_jobs=-1):
    n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
    if max_cover is None:
        max_cover = float('inf')
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    open = openers[compressor]
    
    with open(filename, 'rb') as init:
        counts_d = dill.load(init)['counts']
    res = dict()
    left = float('inf')
    for c in counts_d.values():
        left = min(c[:, (0, 1)].min(), left)
    left -= 1
    if beta:
        stat_fun = calc_stats_betabinom
        params = est_binom_params(counts_d, left, n_jobs=n_jobs, w=w, est_p=estimate_p, dist='betabinom', max_cover=max_cover, inv_kl=inv_kl)
    else:
        stat_fun = calc_stats_binom
        params = est_binom_params(counts_d, left, n_jobs=n_jobs, w=w, est_p=estimate_p, dist='binom', max_cover=max_cover, inv_kl=inv_kl)
    print(name, inv_kl, params)
    for bad in counts_d:
        for allele in ('ref', 'alt'):
            sub_res = dict()
            if allele not in res:
                res[allele] = dict()
            res[allele][bad] = sub_res
            swap = allele != 'ref'        
            counts = counts_d[bad][:, (1, 0) if swap else (0, 1)].copy()
            alt = counts[:, 1]
            sub_c = [(u, counts[alt == u, 0]) for u in np.unique(alt)]
            chunksize = int(np.ceil(len(sub_c) / n_jobs))
            with Pool(n_jobs) as p:
                f = partial(stat_fun, w=w, bad=bad, left=left, params=params, swap=swap)
                it = p.imap_unordered(f, sub_c, chunksize=chunksize) if n_jobs > 1 else map(f, sub_c)
                for r in it:
                    sub_res.update(r)
    filename = f'{name}.comb.{compressor}'
    if os.path.isfile(filename):
        os.remove(filename)
    with open(f'{name}.test.{compressor}', 'wb') as f:
        dill.dump(res, f)
    return res, params
