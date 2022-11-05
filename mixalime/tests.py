#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from .utils import get_init_file, dictify_params, get_model_creator, openers
from betanegbinfit.utils import get_params_at_slice
from multiprocessing import cpu_count, Pool
from functools import partial
import numpy as np
import dill
import os


def calc_stats(t: tuple, inst_params: dict, params: dict, swap: bool,
               correction: str = None, gof_tr=None, max_size=None):
    alt, counts, rmsea = t
    res = list()
    params = get_params_at_slice(params, alt, clip_at_max_slice=False)
    if gof_tr is not None and rmsea.get(alt, np.inf) > gof_tr:
        params['r'] = alt
        if 'k' in params:
            inst_params = inst_params.copy()
            inst_params['dist'] = 'NB'
        if 'w' in params:
            params['w'] = 1.0
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
        cdf_l, cdf_r, c = it[:3]
        cdf = w * cdf_l + (1 - w) * cdf_r
        res.append((float(1 - cdf), np.log2(c) - np.log2(mean), ))
    if swap:
        res = {(alt, c): v for c, v in zip(counts, res)}
    else:
        res = {(c, alt): v for c, v in zip(counts, res)}
    return res



def test(name: str, correction: str = None, gof_tr: float = None, n_jobs: int = -1):
    n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    open = openers[compressor]
    
    with open(f'{name}.fit.{compressor}', 'rb') as fit,  open(filename, 'r') as init:
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
            sub_c = [(u, counts[alt == u, 0], st.get(u, {'rsmea': np.nan})) for u in np.unique(alt)]
            max_size = max(map(lambda x: len(x[1]), sub_c))
            chunksize = int(np.ceil(len(sub_c) / n_jobs))
            with Pool(n_jobs) as p:
                f = partial(calc_stats, inst_params=inst_params, params=params, gof_tr=gof_tr, correction=correction, swap=swap,
                            max_size=max_size)
                for r in p.imap_unordered(f, sub_c, chunksize=chunksize):
                    sub_res.update(r)
    filename = f'{name}.comb.{compressor}'
    if os.path.isfile(filename):
        os.remove(filename)
    with open(f'{name}.test.{compressor}', 'wb') as f:
        dill.dump(res, f)
    return res
