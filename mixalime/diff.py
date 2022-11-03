#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .utils import get_init_file, get_model_creator, dictify_params, parse_filenames, openers
from multiprocessing import Pool, cpu_count
from statsmodels.stats import multitest
from collections import defaultdict
from scipy.stats import chi2
from functools import partial
from copy import deepcopy
import pandas as pd
import numpy as np
import dill
import os


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

def get_closest_param(params: dict, slc: float, model_name: str):
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
    return res


def difftest(counts: tuple[tuple, np.ndarray, np.ndarray, np.ndarray],
             inst_params: dict, params: dict, skip_failures=False, max_sz=None, bad=1.0):
    if not hasattr(difftest, '_cache'):
        difftest._cache = dict()
    snv, counts_a, counts_b, counts = counts
    cache = difftest._cache
    key = tuple(inst_params.values())
    name = inst_params['name']
    if key not in cache:
        inst_params = deepcopy(inst_params)
        inst_params['name'] = 'line_diff'
        inst_params['estimate_p'] = True
        model = get_model_creator(**inst_params)()
        x0 = np.zeros(model.num_params, dtype=float)
        model.override_start([x0])
        cache[key] = model, x0
    else:
        model, x0 = cache[key]
    if max_sz is not None:
        model.mask = np.zeros(max_sz, dtype=bool)
    res = list()
    for allele in ('ref', 'alt'):
        if allele == 'alt':
            counts_a = counts_a[:, (1, 0, 2)]; counts_b = counts_b[:, (1, 0, 2)]; counts = counts[:, (1, 0, 2)]
        ps = get_closest_param(params[allele], counts[:, 1].mean(), name)
        ps['p1'] = 1 / (bad + 1)
        for p, v in ps.items():
            model.set_param(p, x0, v)
        try:
            unified = model.fit(counts, use_prev=False, )
            a = model.fit(counts_a, use_prev=False, )
            b = model.fit(counts_b, use_prev=False, )
            success = a['success'] & b['success'] & unified['success']
            if not success and skip_failures:
                return None, snv
        except (KeyError, IndexError):
            if skip_failures:
                return None, snv
            res.append((float('nan'), float('nan'), float('nan'), float('nan')))
        lrt = -2 * (unified['loglik'] - (a['loglik'] + b['loglik']))
        pval = chi2.sf(lrt, 1)
        res.append((pval, unified['p1'], a['p1'], b['p1']))
    return res, snv
        
        
def differential_test(name: str, group_a: list[str], group_b: list[str], min_samples=2, min_cover=0,
                      max_cover=np.inf, skip_failures=True, test_groups=True, alpha=0.05, subname=None, n_jobs=-1):  
    if max_cover is None:
        max_cover = np.inf
    if min_cover is None:
        min_cover = np.inf
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
    snvs_a = {k: snvs_a[k] for k in snvs}
    snvs_b = {k: snvs_b[k] for k in snvs}
    with open(f'{name}.fit.{compressor}', 'rb') as f:
        fits = dill.load(f)
    _counts_a, _counts_b, _counts = build_count_tables(snvs_a, snvs_b)
    res = list()
    bad = 1
    counts_a = _counts_a[bad]; counts_b = _counts_b[bad]; counts = _counts[bad]
    params = {'ref': dictify_params(fits['ref'][bad]['params']),
              'alt': dictify_params(fits['ref'][bad]['params'])}
    inst_params = fits['ref'][1]['inst_params']
    cols = ['ref_pval', 'ref_p_ab', 'ref_p_a', 'ref_p_b', 
            'alt_pval', 'alt_p_ab', 'alt_p_a', 'alt_p_b']
    if test_groups:
        (whole_ref, whole_alt), _ = difftest(('all', counts_a, counts_b, counts), inst_params, params, skip_failures=False, bad=bad)
        df_whole = pd.DataFrame([list(whole_ref) + list(whole_alt)], columns=cols)
        
    counts = [(snv, *[c[bad] for c in build_count_tables({snv: snvs_a[snv]}, {snv: snvs_b[snv]})])
              for snv, it in snvs_a.items() if it[0][0] == bad] 
    max_sz = max(c[-1].shape[0] for c in counts)
    f = partial(difftest, params=params, inst_params=inst_params, skip_failures=skip_failures, max_sz=max_sz, bad=bad)
    chunk_size = len(counts) // n_jobs
    with Pool(n_jobs) as p:
        for r, snv in p.imap_unordered(f, counts, chunksize=chunk_size):
            if r is None:
                continue
            res.append([snv] + list(r[0]) + list(r[1]))

    df = pd.DataFrame(res, columns=['ind'] + cols)
    _, df['ref_fdr_pval'], _, _ = multitest.multipletests(df['ref_pval'], alpha=alpha, method='fdr_bh')
    _, df['alt_fdr_pval'], _, _ = multitest.multipletests(df['alt_pval'], alpha=alpha, method='fdr_bh')
    
    res = {subname: {'snvs': df}}
    if test_groups:
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