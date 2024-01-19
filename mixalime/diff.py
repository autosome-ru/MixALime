#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .utils import get_init_file, dictify_params, select_filenames, openers
from betanegbinfit import distributions as dists
from betanegbinfit.models import ModelWindow
from multiprocessing import Pool, cpu_count, get_context
from statsmodels.stats import multitest
from collections import defaultdict
from scipy.stats import chi2, norm
from functools import partial
from copy import deepcopy
from typing import List, Tuple
from scipy.optimize import minimize_scalar
import jax.numpy as jnp
import pandas as pd
import numpy as np
import jax
import dill
import re
import os


class Model():
    def __init__(self, dist: str, left=4, mask_size=0, model_name='window', param_mode='window', r_transform=None, 
                 symmetrify=False, bad=1, max_count=1e6):
        self.dist = dist
        self.left = left
        self.allowed_const = left + 1
        self.mask = np.zeros(mask_size, dtype=bool)
        self.model_name = model_name
        self.param_mode = param_mode
        self.r_transform = r_transform
        self.grad = jax.jit(jax.jacfwd(self.fun, argnums=0))
        self.fim = jax.jit(jax.jacfwd(jax.jacfwd(self.negloglik, argnums=0), argnums=0))
        self.bad = bad
        self.symmetrify = symmetrify
        self.max_count = int(max_count)
    
    @partial(jax.jit, static_argnums=(0, ))
    def fun(self, p: float, r: jax.numpy.ndarray, k: jax.numpy.ndarray, data: jax.numpy.ndarray,  w:jax.numpy.ndarray,
            mask: jax.numpy.ndarray):
        left = self.left
        if self.dist == 'NB':
            logl = dists.LeftTruncatedNB.logprob(data, r, p, left, r_transform=self.r_transform)
        elif self.dist == 'MCNB':
            logl = dists.LeftTruncatedMCNB.logprob_recurrent(data, r, p, left,
                                                             max_sz=self.max_count + 1, r_transform=self.r_transform)
            logl = jnp.take_along_axis(logl, data.reshape(-1, 1), axis=1).flatten()
        else:
            logl = dists.LeftTruncatedBetaNB.logprob(data, p, k, r, left, r_transform=self.r_transform)
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
            res = dists.LeftTruncatedBetaNB.sample(p, k, r, left, size=(len(r),), r_transform=self.r_transform)
        else:
            res = dists.LeftTruncatedNB.sample(r, p, left, size=(len(r),), r_transform=self.r_transform)
        res = np.stack([res, alt]).T 
        res, c = np.unique(res, axis=0, return_counts=True)
        res = np.append(res, c.reshape(-1, 1), axis=1)
        return res
    
    def minimize_scalar(self, f, xatol=1e-16, steps=10):
        if steps:
            ps = list(np.linspace(0.01, 0.99, steps))
            i = np.nanargmin(list(map(f, ps)))
            ps = [0.0] + ps + [1.0]
            b = ps[i], ps[i + 2]
        else:
            b = (0.0, 1.0)
        return minimize_scalar(f, bounds=b, method='bounded', options={'xatol': xatol, 'maxiter': 200})
    
    def adjust_r(self, r, k, w=None):
        bad = self.bad
        if w is None or bad == 1:
            return r
        p2 = 1 / (bad + 1)
        p1 = 1 - p2
        left = self.left
        if self.dist == 'NB':
            mean = lambda p: dists.LeftTruncatedNB.mean(r, p, left, r_transform=self.r_transform)
        elif self.dist == 'MCNB':
            mean = lambda p: dists.LeftTruncatedMCNB.mean(r, p, left, r_transform=self.r_transform)
        else:
            mean = lambda p: dists.LeftTruncatedBetaNB.mean(p, k, r, left, r_transform=self.r_transform)
        m1 = mean(p1)
        m2 = mean(p2)
        correction = m1 / (w * m1 + (1.0 - w) * m2)
        return r * correction
        

    def fit(self, data: np.ndarray, params: dict, compute_var=True, sandwich=True, n_bootstrap=0):
        name = self.model_name
        if self.symmetrify:
            data = ModelWindow.symmetrify_counts(data)
        data, fixed, w = data.T
        n = len(data)
        if self.param_mode == 'window' or self.model_name == 'slices':
            r = np.zeros(n, dtype=float)
            k = np.zeros(n, dtype=float)
            for i in range(n):
                slc = fixed[i]
                ps = get_closest_param(params, slc, name, True)
                r[i] = ps['r']
                if self.dist == 'BetaNB':
                    k[i] = ps.get('k', 6e3)
        else:
            slc = fixed.mean()
            ps = get_closest_param(params, slc, name, False)
            r = ps['mu'] + ps['b'] * fixed ** ps.get('slice_power', 1.0)
            if 'mu_k' in ps:
                k = ps['mu_k'] + ps['b_k'] * np.log(fixed)
            else:
                k = np.zeros(len(r))
            r = self.adjust_r(r, k, ps.get('w', None))
        data, r, k, w = self.update_mask(data, r, k, w)
        mask = self.mask
        f = partial(self.negloglik, r=r, k=k, data=data, w=w, mask=mask)
        grad_w = partial(self.grad, r=r, k=k, data=data, w=w, mask=mask)
        fim = partial(self.fim, r=r, k=k, data=data, w=w, mask=mask)
        res = self.minimize_scalar(f)
        x = res.x
        bs = list()
        for i in range(n_bootstrap):
            r_, k_, w_ =  r[:n], k[:n], w[:n]
            data, _, w_ = self.sample(w_, fixed[:n], x, r_, k_).T
            data, r_, k_, w_ = self.update_mask(data, r_, k_, w_)
            mask = self.mask
            f = partial(self.negloglik, r=r_, k=k_, data=data, w=w_, mask=mask)
            res_ = self.minimize_scalar(f)
            if res_.success:
                bs.append(res_.x) 

        res.x = float(res.x)
        correction = res.x - np.mean(bs) if n_bootstrap else 0.0
        res.x = np.clip(res.x + correction, 1e-12, 1.0 - 1e-12)
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
        if not t:
            continue
        if len(t) >= min_samples and max(sum(v[1:]) for v in t) >= min_cover:
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
        res[bad] = np.array(r, dtype=int)
    return res

# def build_count_tables(snvs_a: dict, snvs_b: dict):
#     counts_a = build_count_table(snvs_a)
#     counts_b = build_count_table(snvs_b)
#     counts = deepcopy(counts_a)
#     for bad in counts:
#         c = counts[bad]
#         cb = counts_b[bad]
#         for t, n in cb.items():
#             c[t] += n
#     return map(count_dict_to_numpy, (counts_a, counts_b, counts))

def build_count_tables(*snvs):
    counts = list(map(build_count_table, snvs))
    bads = set()
    for t in counts:
        bads |= set(t)
    base = dict()
    for bad in bads:
        c = defaultdict(int)
        for count in counts:
            cb = count[bad]
            for t, n in cb.items():
                c[t] += n
        base[bad] = c
    counts.append(base)
    return list(map(count_dict_to_numpy, counts))

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

def lrt_test(counts: tuple,
        inst_params: dict, params: dict, skip_failures=False, max_sz=None, bad=1.0,
        param_mode='window', n_bootstrap=0, max_count=1e5):
    if not hasattr(lrt_test, '_cache'):
        lrt_test._cache = dict()
    max_count = int(max_count)
    snv = counts[0]
    counts = counts[1:]
    cache = lrt_test._cache
    if max_sz is None:
        max_sz = 0
    if type(max_sz) is not int:
        key = (inst_params['dist'], inst_params['left'], max_sz[0], inst_params['name'], param_mode,
               inst_params['r_transform'], inst_params['symmetrify'])
        key_long = (inst_params['dist'], inst_params['left'], max_sz[1], inst_params['name'], param_mode,
                    inst_params['r_transform'], inst_params['symmetrify'])
    else:
        key = (inst_params['dist'], inst_params['left'], max_sz if max_sz else 0, inst_params['name'], param_mode,
               inst_params['r_transform'], inst_params['symmetrify'])
    if key not in cache:
        model = Model(*key, bad=bad, max_count=max_count)
        if type(max_sz) is not int:
            model_long = Model(*key_long, bad=bad, max_count=max_count)
            cache[key] = (model, model_long)
        else:
            cache[key] = model
            model_long - model
    else:
        if type(max_sz) is not int:
            model, model_long = cache[key]
        else:
            model = cache[key]
            model_long = model
    res = list()
    for allele in ('ref', 'alt'):
        if allele == 'alt':
            counts = [c[:, (1, 0, 2)] for c in counts]
        try:
            ps = list()
            loglik = 0
            for count in counts[:-1]:
                r, logl = model.fit(count, params[allele], False, n_bootstrap=n_bootstrap)
                if not r.success and not skip_failures:
                    return None, snv
                ps.append(r.x)
                loglik -= logl
            r, logl = model_long.fit(counts[-1], params[allele], False, n_bootstrap=n_bootstrap)
            ps.append(r.x)
            loglik += logl
        except (KeyError, IndexError):
            if skip_failures:
                return None, snv
            res.append((float('nan'), float('nan'), float('nan'), float('nan')))
        lrt = 2 * loglik
        pval = chi2.sf(lrt, len(counts) - 2)
        res.append([pval] + ps)
    n = [cnt[:, -1].sum() for cnt in counts][:-1]
    return res, [snv] + n

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
    key = (inst_params['dist'], inst_params['left'], max_sz if max_sz else 0, inst_params['name'], param_mode,
           inst_params['r_transform'], inst_params['symmetrify'])
    if key not in cache:
        model = Model(*key, bad=bad)
        cache[key] = model
    else:
        model = cache[key]
    res = list()
    for allele in ('ref', 'alt'):
        if allele == 'alt':
            counts_a = counts_a[:, (1, 0, 2)]; counts_b = counts_b[:, (1, 0, 2)]; counts = counts[:, (1, 0, 2)]
        try:
            a_r, a_var = model.fit(counts_a, params[allele], sandwich=robust_se, n_bootstrap=n_bootstrap)
            a_var = np.clip(a_var, 0.0, np.inf)
            if (1.0 - a_r.x) ** 2 < a_var:
                a_var = (1.0 - a_r.x) ** 2
            a_p = a_r.x
            b_r, b_var = model.fit(counts_b, params[allele], sandwich=robust_se, n_bootstrap=n_bootstrap)
            b_var = np.clip(b_var, 0.0, np.inf)
            if (1.0 - b_r.x) < b_var ** 0.5:
                b_var = (1.0 - b_r.x) ** 2
            b_p = b_r.x
            correct = (a_var + b_var > 0) & np.isfinite(a_var) & np.isfinite(b_var)
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
            if logit_transform:
                _a_p, _a_var = transform_p(a_p, a_var)
                _b_p, _b_var = transform_p(b_p, b_var)
            else:
                _a_p, _a_var = a_p, a_var
                _b_p, _b_var = b_p, b_var
            stat = abs(a * _a_p + b * _b_p + eq)
            var = a ** 2 * _a_var + b ** 2 * _b_var
            pval = 2 * norm.sf(stat, scale=(var) ** 0.5, loc=0.0)
        res.append((pval, a_p, b_p, a_var ** 0.5, b_var ** 0.5))
    n_a, n_b = counts_a[:, -1].sum(), counts_b[:, -1].sum()
    return res, (snv, n_a, n_b)

def _bad_in(t, bad):
    for t in t[1:]:
        if t[-1] == bad:
            return True
    return False
        
def differential_test(name: str, group_a: List[str], group_b: List[str], mode='wald', min_samples=2, min_cover=0,
                      max_cover=np.inf, skip_failures=True, group_test=True, alpha=0.05, max_cover_group_test=None,
                      filter_chr=None, filter_id=None, contrasts=(1, -1, 0), subname=None, param_mode='window',
                      logit_transform=False, robust_se=True, n_bootstrap=0, fit: str = None, n_jobs=-1):
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
    init_filename = get_init_file(name)
    compressor = init_filename.split('.')[-1]
    open = openers[compressor]
    with open(init_filename, 'rb') as f:
        snvs = dill.load(f)
        scorefiles = snvs['scorefiles']
        snvs = snvs['snvs']
    if type(group_a) is str:
        group_a = [group_a]
    if type(group_b) is str:
        group_b = [group_b]
    group_a = select_filenames(group_a, scorefiles)
    group_b = select_filenames(group_b, scorefiles)
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
    if fit:
        comp_fit = fit.split('.')[-1]
        open2 = openers[comp_fit]
    else:
        fit = f'{name}.fit.{compressor}'
        open2 = open
    with open2(fit, 'rb') as f:
        fits = dill.load(f)
    if group_test:
        _counts_a, _counts_b, _counts = build_count_tables(snvs_a, snvs_b)
    res = list()
    for bad in fits['ref'].keys():
        params = {'ref': dictify_params(fits['ref'][bad]['params']),
                  'alt': dictify_params(fits['ref'][bad]['params'])}
        inst_params = fits['ref'][1]['inst_params']
        if mode == 'lrt':
            cols = ['ref_pval', 'ref_p', 'ref_p_control', 'ref_p_test', 
                    'alt_pval', 'alt_p', 'alt_p_control', 'alt_p_test']
            test_fun = partial(lrt_test, inst_params=inst_params, params=params, skip_failures=False, bad=bad,
                               param_mode=param_mode, n_bootstrap=n_bootstrap)
        else:
            cols = ['ref_pval', 'ref_p_control', 'ref_p_test', 'ref_std_control', 'ref_std_test',
                    'alt_pval', 'alt_p_control', 'alt_p_test', 'alt_std_control', 'alt_std_test']
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
                  for snv, it in snvs_a.items() if _bad_in(it, bad)]
        if not counts:
            continue
        max_sz = max(c[-1].shape[0] for c in counts)
        f = partial(test_fun, skip_failures=skip_failures, max_sz=max_sz)
        chunk_size = len(counts) // n_jobs
        
        with Pool(n_jobs) as p:
            if n_jobs > 1:
                it = p.imap_unordered(f, counts, chunksize=chunk_size)
            else:
                it = map(f, counts)
            for r, t in it:
                if r is None:
                    continue
                res.append([*t] + list(r[0]) + list(r[1]))
    df = pd.DataFrame(res, columns=['ind', 'n_control', 'n_test'] + cols)
    if df.empty:
        df['ref_fdr_pval'] = None
        df['alt_fdr_pval'] = None
    else:
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

def anova_test(name: str, groups: List[str], alpha=0.05, subname=None, fit=None, min_groups=2,
               min_samples=4, min_cover=0, max_cover=np.inf, param_mode='window', skip_failures=True, n_jobs=1):
    if max_cover is None:
        max_cover = np.inf
    if min_cover is None:
        min_cover = np.inf
    n_jobs = cpu_count() if n_jobs == -1 else n_jobs
    init_filename = get_init_file(name)
    compressor = init_filename.split('.')[-1]
    open = openers[compressor]
    with open(init_filename, 'rb') as f:
        snvs = dill.load(f)
        scorefiles = snvs['scorefiles']
        snvs = snvs['snvs']
    groups = [[scorefiles.index(f) for f in select_filenames([pattern], scorefiles)] for pattern in groups]
    snvs_groups = [get_snvs_for_group(snvs, g, min_samples=min_samples) for g in groups]
    snvs_groups_count = defaultdict(int)
    for g in snvs_groups:
        for snv in g:
            snvs_groups_count[snv] += 1
    snvs = {snv for snv, n in snvs_groups_count.items() if n >= min_groups}
    snvs = {snv for i, snv in enumerate(snvs) if i < 100}
    snvs_groups = [{k: g.get(k, list()) for k in snvs} for g in snvs_groups]
    if fit:
        comp_fit = fit.split('.')[-1]
        open2 = openers[comp_fit]
    else:
        fit = f'{name}.fit.{compressor}'
        open2 = open
    with open2(fit, 'rb') as f:
        fits = dill.load(f)
    res = list()
    
    for bad in fits['ref'].keys():
        params = {'ref': dictify_params(fits['ref'][bad]['params']),
                  'alt': dictify_params(fits['ref'][bad]['params'])}
        inst_params = fits['ref'][1]['inst_params']
        cols = ['ref_pval', 'ref_p', 'ref_p_control', 'ref_p_test', 
                'alt_pval', 'alt_p', 'alt_p_control', 'alt_p_test']
        test_fun = partial(lrt_test, inst_params=inst_params, params=params, skip_failures=False, bad=bad,
                           param_mode=param_mode)
        counts = list()
        group_inds = dict()
        for snv in snvs:
            inds = list()
            it = list()
            r = build_count_tables(*[{snv: g[snv]} for g in snvs_groups])
            for i in range(len(snvs_groups)):
                try:
                    t = r[i][bad]
                    if not t.shape[0]:
                        continue
                    it.append(t)
                except KeyError:
                    continue
                inds.append(i)
            if inds:
                it.append(r[-1][bad])
                counts.append((snv, *it))
                group_inds[snv] = np.array(inds)
        if not counts:
            continue
        max_sz = max(c.shape[0] for tc in counts for c in tc[1:-1])
        max_sz = (max_sz, max(c[-1].shape[0] for c in counts))
        print(max_sz)
        max_count = max(c[:, [0, 1]].max() for tc in counts for c in tc[1:-1])
        max_count = 100000
        f = partial(test_fun, skip_failures=skip_failures, max_sz=max_sz, max_count=max_count)
        chunk_size = len(counts) // n_jobs
        with Pool(n_jobs) as p:
            if n_jobs > 1:
                it = p.map_async(f, counts, chunksize=chunk_size)
                it.wait()
                it = it.get()
            else:
                it = map(f, counts)
        z = np.repeat(np.nan, len(snvs_groups) + 2)
        nz = np.repeat(0, len(snvs_groups))
        N = len(counts)
        K = 0
        for r, t in it:
            if r is None:
                continue
            K += 1
            inds = group_inds[t[0]]
            ref = z.copy()
            ref[inds] = r[0][1:-1]
            ref[-1] = r[0][0]
            ref[-2] = r[0][-1]
            alt = z.copy()
            alt[inds] = r[1][1:-1]
            alt[-1] = r[1][0]
            alt[-2] = r[1][-1]
            n = nz.copy()
            n[inds] = t[1:]
            res.append([t[0]] + list(n) + list(ref) + list(alt))
            # print(K / N, K, N, max_sz, max_count)
    cols = [f'p_{i + 1}' for i in range(len(snvs_groups))] + ['p_all', 'pval']
    cols = ['ind'] + [f'n_{i + 1}' for i in range(len(snvs_groups))] + [f'ref_{c}' for c in cols] + [f'alt_{c}' for c in cols]
    df = pd.DataFrame(res, columns=cols)
    if df.empty:
        df['ref_fdr_pval'] = None
        df['alt_fdr_pval'] = None
    else:
        _, df['ref_fdr_pval'], _, _ = multitest.multipletests(df['ref_pval'], alpha=alpha, method='fdr_bh')
        _, df['alt_fdr_pval'], _, _ = multitest.multipletests(df['alt_pval'], alpha=alpha, method='fdr_bh')
    res = {subname: {'tests': df, 'snvs': snvs_groups}}
    filename = f'{name}.anova.{compressor}'
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            d = dill.load(f)
            if subname in d:
                del d[subname]
            res.update(d)
    with open(filename, 'wb') as f:
        dill.dump(res, f)
    return res