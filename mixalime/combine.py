# -*- coding: utf-8 -*-
from scipy import stats as st
import numpy as np
import dill
import os
import re
from .utils import get_init_file, openers, parse_filenames
from multiprocessing import cpu_count, Pool
from functools import partial
from statsmodels.stats import multitest
from gmpy2 import log, log1p
import logging

    
def combine_p_values_logit(pvalues):
    pvalues = list(filter(lambda x: x  < 1, pvalues))
    k = len(pvalues)
    if k == 0:
        return 1.0
    elif k == 1:
        return float(pvalues[0])
    statistic = float(-sum(map(log, pvalues)) + sum(map(lambda x: log1p(-x), pvalues)))
    nu = np.int_(5 * k + 4)
    approx_factor = np.sqrt(np.int_(3) * nu / (np.int_(k) * np.square(np.pi) * (nu - np.int_(2))))
    pval = st.distributions.t.sf(statistic * approx_factor, nu)
    if pval == 0:
        statistic = float(-2 * sum(map(log, pvalues)))
        pval = st.distributions.chi2.sf(statistic, 2 * k)
    return pval

def combine_es(es, pvalues):
    pvalues = np.array(list(map(float, pvalues)))
    es = np.array(es)
    weights = -np.log10(pvalues)
    inds = np.isfinite(weights)
    if not inds.sum():
        return np.mean(es)
    weights = weights[inds]
    es = es[inds]
    s = weights.sum()
    if s == 0:
        weights = 1 / len(weights)
    else:
        weights /= s
    return np.sum(weights * es)

def combine_stats(t, stats, groups, min_cnt_sum=20):
    k, lt = t
    bad = lt[0][0]
    lt = lt[1:]
    if groups:
        lt = filter(lambda x: x[0] in groups, lt)
    lt = [t[1:] for t in lt]
    if not lt or max(sum(t) for t in lt) < min_cnt_sum:
        return (np.nan, np.nan), (np.nan, np.nan), None
    ref_pvals, ref_es = zip(*[stats['ref'][bad][t] for t in lt])
    alt_pvals, alt_es = zip(*[stats['alt'][bad][t] for t in lt])
    ref = combine_p_values_logit(ref_pvals)
    alt = combine_p_values_logit(alt_pvals)
    ref_es = combine_es(ref_es, ref_pvals)
    alt_es = combine_es(alt_es, alt_pvals)
    return (ref, alt), (ref_es, alt_es), k
    


def combine(name: str, group_files=None, alpha=0.05, min_cnt_sum=20, filter_id=None, filter_chr=None, subname=None, n_jobs=1, save_to_file=True):
    if group_files is None:
        group_files = list()
    else:
        group_files = parse_filenames(group_files)
    if filter_chr is not None:
        filter_chr = re.compile(filter_chr)
    if filter_id is not None:
        filter_id = re.compile(filter_id)
    n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    open = openers[compressor]
    with open(filename, 'rb') as f:
        init = dill.load(f)
        snvs = init['snvs']
        scorefiles = init['scorefiles']
        del init
    filename = f'{name}.test.{compressor}'
    with open(filename, 'rb') as f:
        stats = dill.load(f)
    groups = set()
    for file in group_files:
        try:
            i = scorefiles.index(file)
            groups.add(i)
        except ValueError:
            logging.error(f'Unknown scorefile {file}')
    ref_comb_pvals = list()
    alt_comb_pvals = list()
    ref_comb_es = list()
    alt_comb_es = list() 
    comb_names = list()
    its = snvs.items()
    if filter_id:
        its = list(filter(lambda x: x[1][0][1] and filter_id.match(x[1][0][1]), its))
    if filter_chr:
        its = list(filter(lambda x: filter_chr.match(x[0]), its))
    with Pool(n_jobs) as p:
        sz = len(its) // n_jobs
        f = partial(combine_stats, stats=stats, groups=groups, min_cnt_sum=min_cnt_sum)
        for (ref, alt), (ref_es, alt_es), k in p.imap_unordered(f, its, chunksize=sz):
            if k is None:
                continue
            ref_comb_pvals.append(ref)
            alt_comb_pvals.append(alt)
            ref_comb_es.append(ref_es)
            alt_comb_es.append(alt_es)
            comb_names.append(k)
    ref_comb_pvals = np.array(ref_comb_pvals)
    inds = ref_comb_pvals == 0.0
    if np.any(inds):
        ref_comb_pvals[inds] = ref_comb_pvals[~inds].min()
    alt_comb_pvals = np.array(alt_comb_pvals)
    inds = alt_comb_pvals == 0.0
    if np.any(inds):
        alt_comb_pvals[inds] = alt_comb_pvals[~inds].min()
    _, ref_fdr_pvals, _, _ = multitest.multipletests(ref_comb_pvals, alpha=alpha, method='fdr_bh')
    _, alt_fdr_pvals, _, _ = multitest.multipletests(alt_comb_pvals, alpha=alpha, method='fdr_bh')
    res = dict()
    for i in range(len(comb_names)):
        
        res[comb_names[i]] = ((ref_comb_pvals[i], alt_comb_pvals[i]), 
                              (ref_comb_es[i], alt_comb_es[i]),
                              (ref_fdr_pvals[i], alt_fdr_pvals[i]))
    res = {subname: res}
    if os.path.isfile(f'{name}.comb.{compressor}'):
        with open(f'{name}.comb.{compressor}', 'rb') as f:
            r = dill.load(f)
            for t in r:
                if t not in res:
                    res[t] = r[t]
    if save_to_file:
        with open(f'{name}.comb.{compressor}', 'wb') as f:
            dill.dump(res, f)
    return res
