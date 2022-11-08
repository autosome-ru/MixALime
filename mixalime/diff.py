#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .utils import get_init_file, dictify_params, parse_filenames, openers
from betanegbinfit import distributions as dists
from multiprocessing import Pool, cpu_count
from statsmodels.stats import multitest
from collections import defaultdict
from scipy.stats import chi2, norm
from functools import partial
from copy import deepcopy
from typing import List, Tuple
from scipy.optimize import minimize_scalar, minimize
import pandas as pd
import numpy as np
import jax
import dill
import re
import os


class Model():
    def __init__(self, dist: str, left=4, mask_size=0, model_name='window', param_mode='window'):
        self.dist = dist
        self.left = left
        self.allowed_const = left + 1
        self.mask = np.zeros(mask_size, dtype=bool)
        self.model_name = model_name
        self.param_mode = param_mode
        self.grad = jax.jit(jax.jacfwd(self.fun, argnums=0))
        self.fim = jax.jit(jax.grad(jax.grad(self.negloglik, argnums=0), argnums=0))
    
    @partial(jax.jit, static_argnums=(0, ))
    def fun(self, p: float, r: jax.numpy.ndarray, k: jax.numpy.ndarray, data: jax.numpy.ndarray,  w:jax.numpy.ndarray,
            mask: jax.numpy.ndarray):
        left = self.left
        if self.dist == 'NB':
            logl = dists.LeftTruncatedNB.logprob(data, r, p, left)
        else:
            logl = dists.LeftTruncatedBetaNB.logprob(data, p, k, r, left)
        logl *= w
        return jax.numpy.where(mask, 0.0, logl)
    
    @partial(jax.jit, static_argnums=(0, ))
    def negloglik(self, p: float, r: jax.numpy.ndarray, k: jax.numpy.ndarray, data: jax.numpy.ndarray,  w:jax.numpy.ndarray,
            mask: jax.numpy.ndarray):
        return -self.fun(p, r, k, data, w, mask).sum()
    
    def update_mask(self, data, r, k, w):
        mask = self.mask
        n = max(len(r), len(data))
        m = len(mask)
        if n > m:
            mask = np.zeros(n, dtype=bool)
            self.mask = mask
            m = n
        mask[:n] = False
        mask[n:] = True
        c = self.allowed_const
        v = max(0, m - len(data)); data = np.pad(data, (0, v), constant_values=c); 
        v = max(0, m - len(w)); w = np.pad(w, (0, v), constant_values=c);
        v = max(0, m - len(r)); r = np.pad(r, (0, v), constant_values=c)
        v = max(0, m - len(k)); k = np.pad(k, (0, v), constant_values=c)
        return data, r, k, w
    
    def sample(self, n, alt, p, r, k):
        n = n.astype(int)
        alt = np.repeat(alt, n)
        r = np.repeat(r, n)
        k = np.repeat(k, n)
        left = self.left
        if self.dist == 'BetaNB':
            res = dists.LeftTruncatedBetaNB.sample(p, k, r, left, size=(len(r), ))
        else:
            res = dists.LeftTruncatedNB.sample(r, p, left, size=(len(r), ))
        res = np.stack([res, alt]).T 
        res, c = np.unique(res, axis=0, return_counts=True)
        res = np.append(res, c.reshape(-1, 1), axis=1)
        return res

    def fit(self, data: np.ndarray, params: dict, compute_var=True, sandwich=True, n_bootstrap=100):
        name = self.model_name
        data, fixed, w = data.T
        n = len(data)
        if self.param_mode == 'window' or self.model_name == 'slices':
            r = np.zeros(n, dtype=float)
            k = np.zeros(n, dtype=float)
            for i in range(n):
                slc = fixed[i]
                ps = get_closest_param(params, slc, name, True)
                r[i] = ps['r']
                if self.dist == 'BetaNB' and 'k' in ps:
                    k[i] = ps['k']
        else:
            slc = fixed.mean()
            ps = get_closest_param(params, slc, name, False)
            r = ps['mu'] + ps['b'] * fixed ** ps.get('slice_power', 1.0)
            if 'mu_k' in ps:
                k = ps['mu_k'] + ps['b_k'] * np.log(fixed)
            else:
                k = np.zeros(len(r))
        data, r, k, w = self.update_mask(data, r, k, w)
        mask = self.mask
        f = partial(self.negloglik, r=r, k=k, data=data, w=w, mask=mask)
        grad_w = partial(self.grad, r=r, k=k, data=data, w=w, mask=mask)
        fim = partial(self.fim, r=r, k=k, data=data, w=w, mask=mask)
        res = minimize_scalar(f, bounds=(0.0, 1.0), method='bounded')
        x = res.x
        bs = list()
        for i in range(n_bootstrap):
            r_, k_, w_ =  r[:n], k[:n], w[:n]
            data, _, w_ = self.sample(w_, fixed[:n], x, r_, k_).T
            data, r_, k_, w_ = self.update_mask(data, r_, k_, w_)
            mask = self.mask
            f = partial(self.negloglik, r=r_, k=k_, data=data, w=w_, mask=mask)
            res_ = minimize_scalar(f, bounds=(0.0, 1.0), method='bounded')
            if res_.success:
                bs.append(res_.x) 

        res.x = float(res.x)
        correction = res.x - np.mean(bs) if n_bootstrap else 0.0
        res.x = res.x + correction
        lf = float(f(res.x))
        if compute_var:
            if sandwich:
                g = grad_w(res.x)
                v = (g ** 2 / w).sum()
                fim = fim(res.x)
                s = -1 if fim < -1e-9 else 1
                return res, s * float(1 / fim ** 2 * v)
            else:
                return res, float(1 / fim(res.x))
        return res, lf
        
    

def get_snvs_for_group(snvs: dict, group: set, min_samples: int = 0, min_cover : int = 0, max_cover : int = np.inf):
    res = dict()
    for k, lt in snvs.items():
        t = list(filter(lambda x: (x[0] in group) and sum(x[1:]) < max_cover, lt[1:]))
        if len(t) >= min_samples and max(sum(v[1:]) for v in t) >= min_cover:
            res[k] = [lt[0]] + t
    return res

def build_count_table(snvs: dict):
    count = defaultdict(lambda: defaultdict(int))
    for lt in snvs.values():
        bad = lt[0][0]
        c = count[bad]
        for t in lt[1:]:
            c[tuple(t[1:])] += 1
    return count

def count_dict_to_numpy(counts: dict):
    res = dict()
    for bad, counts in counts.items():
        r = list()
        for t in sorted(counts):
            r.append(list(t) + [counts[t]])
        res[bad] = np.array(r, dtype=int)
    return res

def build_count_tables(snvs_a: dict, snvs_b: dict):
    counts_a = build_count_table(snvs_a)
    counts_b = build_count_table(snvs_b)
    counts = deepcopy(counts_a)
    for bad in counts:
        c = counts[bad]
        cb = counts_b[bad]
        for t, n in cb.items():
            c[t] += n
    return map(count_dict_to_numpy, (counts_a, counts_b, counts))

def get_closest_param(params: dict, slc: float, model_name: str, compute_line=False):
    res = dict()
    if model_name == 'line':
        ps = str()
    else:
        if f'mu{round(slc)}' in params:
            ps = round(slc)
        else:
            s = 'mu' if model_name == 'window' else 'r'
            n = len(s)
            ps = int(min(filter(lambda x: x.startswith(s) and x[n].isdigit(), params), key=lambda x: abs(int(x[n:]) - slc))[n:])    
    if f'mu{ps}' in params:
        res['mu'] = params[f'mu{ps}']
    if f'b{ps}' in params:
        res['b'] = params[f'b{ps}']
    if f'mu_k{ps}' in params:
        res['mu_k'] = params[f'mu_k{ps}']
    if f'b_k{ps}' in params:
        res['b_k'] = params[f'b_k{ps}']
    if f'r{ps}' in params:
        res['r'] = params[f'k{ps}']
    if f'k{ps}' in params:
        res['k'] = params[f'k{ps}']
    if compute_line:
        if 'mu' in res:
            res['r'] = res['mu'] + res.get('b', 0.0) * slc
        if 'mu_k' in res:
            res['k'] = res['mu_k'] + res.get('b_k', 0.0) * slc
    return res


def lrt_test(counts: Tuple[tuple, np.ndarray, np.ndarray, np.ndarray],
        inst_params: dict, params: dict, skip_failures=False, max_sz=None, bad=1.0,
        param_mode='window'):
    if not hasattr(lrt_test, '_cache'):
        lrt_test._cache = dict()
    snv, counts_a, counts_b, counts = counts
    cache = lrt_test._cache
    key = (inst_params['dist'], inst_params['left'], max_sz if max_sz else 0, inst_params['name'], param_mode)
    if key not in cache:
        model = Model(*key)
        cache[key] = model
    else:
        model = cache[key]
    res = list()
    for allele in ('ref', 'alt'):
        if allele == 'alt':
            counts_a = counts_a[:, (1, 0, 2)]; counts_b = counts_b[:, (1, 0, 2)]; counts = counts[:, (1, 0, 2)]
        try:
            ab_r, ab_logl = model.fit(counts, params[allele], False)
            ab_p = ab_r.x
            a_r, a_logl = model.fit(counts_a, params[allele], False)
            a_p = a_r.x
            b_r, b_logl = model.fit(counts_b, params[allele], False)
            b_p = b_r.x
            success = a_r.success & b_r.success & ab_r.success
            if not success and skip_failures:
                return None, snv
        except (KeyError, IndexError):
            if skip_failures:
                return None, snv
            res.append((float('nan'), float('nan'), float('nan'), float('nan')))
        lrt = -2 * (ab_logl - (a_logl + b_logl))
        pval = chi2.sf(lrt, 1)
        res.append((pval, ab_p, a_p, b_p))
    n_a, n_b = counts_a[:, -1].sum(), counts_b[:, -1].sum()
    return res, (snv, n_a, n_b)

def calc_var(m, data: np.ndarray, renormalize=None):
    data, weights = data[:, :-1], data[:, -1]
    if renormalize:
        n = weights.sum()
    data, weights, mask = m.update_mask(data, weights)
    var = 1 / m.calc_fim(m.last_result.x, data=data, mask=mask,
                        weights=weights)[0][0][-1, -1]
    if renormalize:
        var *= n / renormalize
    return float(var)


def transform_p(p, var):
    pt = np.log(p) - np.log1p(-p)
    var = var * (1 / p + 1 / (1 - p)) ** 2
    return pt, var

def wald_test(counts: Tuple[tuple, np.ndarray, np.ndarray, np.ndarray],
              inst_params: dict, params: dict, skip_failures=False, max_sz=None, bad=1.0,
              contrasts: Tuple[float, float, float] = (1, -1, 0), logit_transform=False,
              param_mode='window', robust_se=True, n_bootstrap=0):
    if not hasattr(wald_test, '_cache'):
        wald_test._cache = dict()
    snv, counts_a, counts_b, counts = counts
    cache = wald_test._cache
    key = (inst_params['dist'], inst_params['left'], max_sz if max_sz else 0, inst_params['name'], param_mode)
    if key not in cache:
        model = Model(*key)
        cache[key] = model
    else:
        model = cache[key]
    res = list()
    for allele in ('ref', 'alt'):
        if allele == 'alt':
            counts_a = counts_a[:, (1, 0, 2)]; counts_b = counts_b[:, (1, 0, 2)]; counts = counts[:, (1, 0, 2)]
        try:
            a_r, a_var = model.fit(counts_a, params[allele], sandwich=robust_se, n_bootstrap=n_bootstrap)
            a_p = a_r.x
            b_r, b_var = model.fit(counts_b, params[allele], sandwich=robust_se, n_bootstrap=n_bootstrap)
            b_p = b_r.x
            correct = (a_var >= 0) & (b_var >= 0) & np.isfinite(a_var) & np.isfinite(b_var)
            if logit_transform:
                a_p, a_var = transform_p(a_p, a_var)
                b_p, b_var = transform_p(b_p, b_var)
            success = a_r.success & b_r.success & correct
            if not success and skip_failures:
                return None, snv
        except (KeyError, IndexError):
            if skip_failures:
                return None, snv
            res.append((float('nan'), float('nan'), float('nan'), float('nan')))
            continue
        if not correct:
            pval = 1.0
        else:
            a, b, eq = contrasts
            stat = abs(a * a_p + b * b_p + eq)
            var = a ** 2 * a_var + b ** 2 * b_var
            pval = 2 * norm.sf(stat, scale=(var) ** 0.5, loc=0.0)
        res.append((pval, a_p, b_p, a_var ** 0.5, b_var ** 0.5))
    n_a, n_b = counts_a[:, -1].sum(), counts_b[:, -1].sum()
    return res, (snv, n_a, n_b)
        
def differential_test(name: str, group_a: List[str], group_b: List[str], mode='wald', min_samples=2, min_cover=0,
                      max_cover=np.inf, skip_failures=True, group_test=True, alpha=0.05, max_cover_group_test=None,
                      filter_chr=None, filter_id=None, contrasts=(1, -1, 0), subname=None, param_mode='window',
                      logit_transform=False, robust_se=True, n_bootstrap=0, n_jobs=-1):
    if max_cover is None:
        max_cover = np.inf
    if min_cover is None:
        min_cover = np.inf
    if max_cover_group_test is None:
        max_cover_group_test = max_cover
    if filter_chr is not None:
        filter_chr = re.compile(filter_chr)
    if filter_id is not None:
        filter_id = re.compile(filter_id)
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    group_a = parse_filenames(group_a)
    group_b = parse_filenames(group_b)
    init_filename = get_init_file(name)
    compressor = init_filename.split('.')[-1]
    open = openers[compressor]
    with open(init_filename, 'rb') as f:
        snvs = dill.load(f)
        scorefiles = snvs['scorefiles']
        snvs = snvs['snvs']
    group_a = {scorefiles.index(f) for f in group_a}
    group_b = {scorefiles.index(f) for f in group_b}
    assert not (group_a & group_b), 'Groups should not intersect.'
    snvs_a = get_snvs_for_group(snvs, group_a, min_samples=min_samples)
    snvs_b = get_snvs_for_group(snvs, group_b, min_samples=min_samples)
    snvs = set(snvs_a) & set(snvs_b)
    if filter_id:
        snvs = set(filter(lambda x: snvs_a[x][0][1] and filter_id.match(x[1][0][1]), snvs))
    if filter_chr:
        snvs = set(filter(lambda x: filter_chr.match(x[0]), snvs))
    snvs_a = {k: snvs_a[k] for k in snvs}
    snvs_b = {k: snvs_b[k] for k in snvs}
    with open(f'{name}.fit.{compressor}', 'rb') as f:
        fits = dill.load(f)
    if group_test:
        _counts_a, _counts_b, _counts = build_count_tables(snvs_a, snvs_b)
    res = list()
    bad = 1
    params = {'ref': dictify_params(fits['ref'][bad]['params']),
              'alt': dictify_params(fits['ref'][bad]['params'])}
    inst_params = fits['ref'][1]['inst_params']
    if mode == 'lrt':
        cols = ['ref_pval', 'ref_p_ab', 'ref_p_a', 'ref_p_b', 
                'alt_pval', 'alt_p_ab', 'alt_p_a', 'alt_p_b']
        test_fun = partial(lrt_test, inst_params=inst_params, params=params, skip_failures=False, bad=bad,
                           param_mode=param_mode)
    else:
        cols = ['ref_pval', 'ref_p_a', 'ref_p_b', 'ref_std_a', 'ref_std_b',
                'alt_pval', 'alt_p_a', 'alt_p_b', 'alt_std_a', 'alt_std_b']
        test_fun = partial(wald_test, inst_params=inst_params, params=params, skip_failures=False, bad=bad,
                           contrasts=contrasts, param_mode=param_mode, logit_transform=logit_transform,
                           robust_se=robust_se, n_bootstrap=n_bootstrap)
    if group_test:
        counts_a = _counts_a[bad]; counts_b = _counts_b[bad]; counts = _counts[bad]
        counts_a = counts_a[counts_a[:, 0] + counts_a[:, 1] < max_cover_group_test]
        counts_b = counts_b[counts_b[:, 0] + counts_b[:, 1] < max_cover_group_test]
        counts = counts[counts[:, 0] + counts[:, 1] < max_cover_group_test]
        (whole_ref, whole_alt), _ = test_fun(('all', counts_a, counts_b, counts))
        df_whole = pd.DataFrame([list(whole_ref) + list(whole_alt)], columns=cols)
    
    counts = [(snv, *[c[bad] for c in build_count_tables({snv: snvs_a[snv]}, {snv: snvs_b[snv]})])
              for snv, it in snvs_a.items() if it[0][0] == bad] 
    max_sz = max(c[-1].shape[0] for c in counts)
    f = partial(test_fun, skip_failures=skip_failures, max_sz=max_sz)
    chunk_size = len(counts) // n_jobs
    with Pool(n_jobs) as p:
        for r, t in p.imap_unordered(f, counts, chunksize=chunk_size):
            if r is None:
                continue
            res.append([*t] + list(r[0]) + list(r[1]))
    df = pd.DataFrame(res, columns=['ind', 'n_a', 'n_b'] + cols)
    _, df['ref_fdr_pval'], _, _ = multitest.multipletests(df['ref_pval'], alpha=alpha, method='fdr_bh')
    _, df['alt_fdr_pval'], _, _ = multitest.multipletests(df['alt_pval'], alpha=alpha, method='fdr_bh')
    
    res = {subname: {'tests': df, 'snvs': (snvs_a, snvs_b)}}
    if group_test:
        res[subname]['whole'] = df_whole
    filename = f'{name}.difftest.{compressor}'
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            d = dill.load(f)
            if subname in d:
                del d[subname]
            res.update(d)
    with open(filename, 'wb') as f:
        dill.dump(res, f)
    return res