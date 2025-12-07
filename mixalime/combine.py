# -*- coding: utf-8 -*-
from scipy import stats as st
import numpy as np
import dill
import os
import re
from .utils import get_init_file, openers, select_filenames
from multiprocessing import cpu_count, Pool, Manager
from collections import defaultdict
from functools import partial
from itertools import islice
from statsmodels.stats import multitest
import logging

    
def combine_p_values_logit(pvalues):
    pvalues = np.array(list(filter(lambda x: x  < 1, pvalues)))
    k = len(pvalues)
    if k == 0:
        return 1.0
    elif k == 1:
        return float(pvalues[0])
    # statistic = float(-sum(map(log, pvalues)) + sum(map(lambda x: log1p(-x), pvalues)))
    statistic = -np.log(pvalues).sum() + np.log1p(-pvalues).sum()
    nu = np.int_(5 * k + 4)
    approx_factor = np.sqrt(np.int_(3) * nu / (np.int_(k) * np.square(np.pi) * (nu - np.int_(2))))
    pval = st.distributions.t.sf(statistic * approx_factor, nu)
    if pval == 0:
        statistic = -2 * np.log(pvalues).sum()
        # statistic = float(-2 * sum(map(log, pvalues)))
        pval = st.distributions.chi2.sf(statistic, 2 * k)
    return pval

def combine_es(es, pvalues, uniform_weights=False):
    pvalues = np.array(list(map(float, pvalues)))
    es = np.array(es)
    weights = -np.log10(pvalues)
    inds = np.isfinite(weights)
    if not inds.sum():
        return np.mean(es)
    weights = weights[inds]
    if uniform_weights:
        weights[:] = 1.0
    es = es[inds]
    s = weights.sum()
    if s == 0:
        weights = 1 / len(weights)
    else:
        weights /= s
    return np.sum(weights * es)

def estimate_min_coverage(test, es_cutoff: float = 1, pval_cutoff: float = 0.5):
    res = defaultdict(dict)
    for allele in test:
        for bad in test[allele]:
            it = test[allele][bad]
            counts = np.array(list(it.keys()))
            coverage = counts.sum(axis=1)
            stop = False
            for cov in np.unique(coverage):
                ind = coverage == cov
                pairs = counts[ind]
                for t in pairs:
                    pvalue, es = it[tuple(t)]
                    if (es > es_cutoff) and (pvalue < pval_cutoff):
                        stop = True
                        break
                if stop:
                    break
            res[allele][bad] = cov if stop else float('inf')
    return res

def combine_stats(inds, snvs, stats, groups, min_cnt_sum=20, uniform_weights=False):
    pvalues = list()
    es = list()
    ks = list()
    for i in inds:
        k, lt = snvs[i]
        lt = lt[1:]
        if groups:
            lt = filter(lambda x: x[0] in groups, lt)
        lt = [t[1:] for t in lt]
        if not lt or not max(sum(t[:-1]) >= min_cnt_sum[t[-1]] for t in lt):
            ref = alt = ref_es = alt_es = np.nan
            k = None
        else:
            ref_pvals, ref_es = zip(*[stats['ref'][t[-1]][t[:-1]] for t in lt])
            alt_pvals, alt_es = zip(*[stats['alt'][t[-1]][t[:-1]] for t in lt])
            ref = combine_p_values_logit(ref_pvals)
            alt = combine_p_values_logit(alt_pvals)
            ref_es = combine_es(ref_es, ref_pvals, uniform_weights=uniform_weights)
            alt_es = combine_es(alt_es, alt_pvals, uniform_weights=uniform_weights)
        pvalues.append((ref, alt))
        es.append((ref_es, alt_es))
        ks.append(k)
    return pvalues, es, ks

def batched(iterable, n):
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

def combine(name: str, group_files=None, alpha=0.05, min_cnt_sum=20, adaptive_min_cover=False, adaptive_es=1.0, adaptive_pval=0.05,
            uniform_weights=False, filter_id=None, filter_chr=None, subname=None, n_jobs=1, save_to_file=True):
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
    if group_files is None:
        group_files = list()
    else:
        group_files = select_filenames(group_files, scorefiles)
        if not group_files:
            raise SyntaxError('No files found for given pattern(s).')
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
    if adaptive_min_cover:
        adaptive_coverage = estimate_min_coverage(stats, es_cutoff=adaptive_es, pval_cutoff=adaptive_pval)
        min_coverage = {bad: min(adaptive_coverage['ref'][bad], adaptive_coverage['alt'][bad]) for bad in adaptive_coverage['ref']}
    else:
        adaptive_coverage = None
        min_coverage = {bad: min_cnt_sum for bad in stats['ref']}
    del scorefiles
    ref_comb_pvals = list()
    alt_comb_pvals = list()
    ref_comb_es = list()
    alt_comb_es = list() 
    comb_names = list()
    its = snvs.items()
    del snvs
    its = list(its)
    if filter_id:
        its = list(filter(lambda x: x[1][0][1] and filter_id.match(x[1][0][1]), its))
    if filter_chr:
        its = list(filter(lambda x: filter_chr.match(x[0]), its))
    with Manager() as manager:
        if n_jobs > 1:
            its = manager.list(its)
        with Pool(n_jobs) as p:
            f = partial(combine_stats, snvs=its, stats=stats, groups=groups, min_cnt_sum=min_coverage,
                        uniform_weights=uniform_weights)
            if n_jobs > 1:
                sz = int(np.ceil(len(its) / n_jobs))
                inds = batched(range(len(its)), sz)
                iterate = p.map(f, inds, chunksize=1)
            else:
                iterate = map(f, [list(range(len(its)))])
            for pvals, es, ks in iterate:
                for (ref, alt), (ref_es, alt_es), k in zip(pvals, es, ks):
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
    res = {'groups': groups, 'snvs': res}
    res = {subname: res}
    if os.path.isfile(f'{name}.comb.{compressor}'):
        with open(f'{name}.comb.{compressor}', 'rb') as f:
            print('eba')
            r = dill.load(f)
            print('gat')
            for t in r:
                if t not in res:
                    res[t] = r[t]
    if save_to_file:
        with open(f'{name}.comb.{compressor}', 'wb') as f:
            dill.dump(res, f)
    return res, adaptive_coverage


def polymbine_stats(inds, snvs, stats_list, groups_list, min_cnt_sums_list, uniform_weights=False):
    """
    Worker function for polymbine. 
    Aggregates statistics across multiple projects for the given indices of SNVs.
    """
    pvalues = list()
    es = list()
    ks = list()
    
    for i in inds:
        k, lt = snvs[i]
        # lt[0] is metadata (id, ref, alt), lt[1:] are replicates
        # Replicates in polymbine have an extra element at the end: project_index
        replicates = lt[1:]
        
        valid_pvals_ref = []
        valid_es_ref = []
        valid_pvals_alt = []
        valid_es_alt = []
        
        has_valid_data = False
        
        for t in replicates:
            # t structure: (file_index, ref_cnt, alt_cnt, ..., bad, project_index)
            p_idx = t[-1]
            bad = t[-2]
            counts = t[:-2] # (file_index, ref, alt, ...)
            file_ind = counts[0]
            cnt_signature = counts[1:] # (ref, alt) usually
            
            # Check if this file is in the allowed groups for this specific project
            if groups_list[p_idx] and file_ind not in groups_list[p_idx]:
                continue

            # Check coverage using the project-specific threshold
            min_cov = min_cnt_sums_list[p_idx].get(bad, 20)
            if sum(cnt_signature) < min_cov:
                continue

            # Retrieve stats from the specific project's dictionary
            # stats structure: stats_list[p_idx]['ref'][bad][(ref, alt)] -> (pval, es)
            try:
                # The original code expects count tuple without file_index for lookup
                lookup_counts = cnt_signature 
                
                # Get Reference Allele Stats
                r_pval, r_es = stats_list[p_idx]['ref'][bad][lookup_counts]
                valid_pvals_ref.append(r_pval)
                valid_es_ref.append(r_es)
                
                # Get Alternative Allele Stats
                a_pval, a_es = stats_list[p_idx]['alt'][bad][lookup_counts]
                valid_pvals_alt.append(a_pval)
                valid_es_alt.append(a_es)
                
                has_valid_data = True
            except KeyError:
                # Should not happen if data integrity is maintained, but safe to skip
                continue

        if not has_valid_data:
            ref = alt = ref_es = alt_es = np.nan
            k = None
        else:
            ref = combine_p_values_logit(valid_pvals_ref)
            alt = combine_p_values_logit(valid_pvals_alt)
            ref_es = combine_es(valid_es_ref, valid_pvals_ref, uniform_weights=uniform_weights)
            alt_es = combine_es(valid_es_alt, valid_pvals_alt, uniform_weights=uniform_weights)
            
        pvalues.append((ref, alt))
        es.append((ref_es, alt_es))
        ks.append(k)
        
    return pvalues, es, ks


def polymbine(output_name: str, names: list[str], group_files=None, alpha=0.05, min_cnt_sum=20, 
              adaptive_min_cover=False, adaptive_es=1.0, adaptive_pval=0.05,
              uniform_weights=False, filter_id=None, filter_chr=None, subname=None, 
              n_jobs=1, save_to_file=True):
    if filter_chr is not None:
        filter_chr = re.compile(filter_chr)
    if filter_id is not None:
        filter_id = re.compile(filter_id)
    n_jobs = cpu_count() - 1 if n_jobs == -1 else n_jobs

    # Lists to store per-project data
    all_stats = []
    all_groups = []
    all_min_coverage = []
    
    # Unified dictionary to store merged SNVs
    # Key: (chr, pos, etc.) -> Value: [metadata, (data_p1...), (data_p2...), ...]
    combined_snvs = defaultdict(lambda: [None]) 

    # 1. Load data from all projects
    for p_idx, name in enumerate(names):
        filename = get_init_file(name)
        if not filename:
            raise FileNotFoundError(f"Init file not found for project: {name}")
            
        compressor = filename.split('.')[-1]
        opener = openers[compressor]
        
        with opener(filename, 'rb') as f:
            init = dill.load(f)
            snvs = init['snvs']
            scorefiles = init['scorefiles']
            del init

        # Determine groups for this project
        current_groups = set()
        if group_files:
            # Check which of the provided group_files belong to this project
            # This logic assumes group_files contains absolute paths or matching patterns
            # that resolve to files present in this project's scorefiles
            selected_files = select_filenames(group_files, scorefiles)
            for file in selected_files:
                try:
                    i = scorefiles.index(file)
                    current_groups.add(i)
                except ValueError:
                    continue 
        else:
            # If no groups specified, all files in the project are valid,
            # but usually empty set implies "all" in existing logic (or explicit logic handles it).
            # Looking at combine_stats: "if groups: lt = filter(...)". 
            # So empty set means "don't filter", i.e., use all.
            pass
            
        all_groups.append(current_groups)

        # Load stats
        filename_test = f'{name}.test.{compressor}'
        with opener(filename_test, 'rb') as f:
            stats = dill.load(f)
        
        # Calculate coverage thresholds
        if adaptive_min_cover:
            adaptive_coverage = estimate_min_coverage(stats, es_cutoff=adaptive_es, pval_cutoff=adaptive_pval)
            min_coverage = {bad: min(adaptive_coverage['ref'][bad], adaptive_coverage['alt'][bad]) for bad in adaptive_coverage['ref']}
        else:
            min_coverage = {bad: min_cnt_sum for bad in stats['ref']}
        
        all_stats.append(stats)
        all_min_coverage.append(min_coverage)
        del scorefiles

        # Merge SNVs
        # We append the project index to the tuple so the worker knows which stats to use
        for k, v in snvs.items():
            if combined_snvs[k][0] is None:
                combined_snvs[k][0] = v[0] # Copy metadata (chr, id, etc) from the first project that has it
            
            # v[1:] are the data tuples: (file_ind, ref, alt, ..., bad)
            # We append p_idx to make it: (file_ind, ref, alt, ..., bad, p_idx)
            for replicate in v[1:]:
                combined_snvs[k].append(replicate + (p_idx,))
        
        del snvs

    # 2. Filter SNVs
    its = combined_snvs.items()
    del combined_snvs
    its = list(its)
    
    if filter_id:
        its = list(filter(lambda x: x[1][0][1] and filter_id.match(x[1][0][1]), its))
    if filter_chr:
        its = list(filter(lambda x: filter_chr.match(x[0]), its))

    # 3. Parallel Processing
    ref_comb_pvals = list()
    alt_comb_pvals = list()
    ref_comb_es = list()
    alt_comb_es = list() 
    comb_names = list()

    with Manager() as manager:
        if n_jobs > 1:
            its = manager.list(its)
            # We treat stats as read-only. Passing large dicts via Manager is slow.
            # Passing them as regular arguments to partial relies on pickle.
            # If stats are huge, this might be memory intensive, but standard for this pattern.
        
        with Pool(n_jobs) as p:
            f = partial(polymbine_stats, snvs=its, stats_list=all_stats, 
                        groups_list=all_groups, min_cnt_sums_list=all_min_coverage,
                        uniform_weights=uniform_weights)
            
            if n_jobs > 1:
                sz = int(np.ceil(len(its) / n_jobs))
                inds_gen = batched(range(len(its)), sz)
                iterate = p.map(f, inds_gen, chunksize=1)
            else:
                iterate = map(f, [list(range(len(its)))])
            
            for pvals, es, ks in iterate:
                for (ref, alt), (ref_es, alt_es), k in zip(pvals, es, ks):
                    if k is None:
                        continue
                    ref_comb_pvals.append(ref)
                    alt_comb_pvals.append(alt)
                    ref_comb_es.append(ref_es)
                    alt_comb_es.append(alt_es)
                    comb_names.append(k)

    # 4. Post-processing (FDR & Saving)
    # The logic below is identical to combine()
    ref_comb_pvals = np.array(ref_comb_pvals)
    inds = ref_comb_pvals == 0.0
    if np.any(inds):
        ref_comb_pvals[inds] = ref_comb_pvals[~inds].min() if np.any(~inds) else 1.0
        
    alt_comb_pvals = np.array(alt_comb_pvals)
    inds = alt_comb_pvals == 0.0
    if np.any(inds):
        alt_comb_pvals[inds] = alt_comb_pvals[~inds].min() if np.any(~inds) else 1.0

    # Handle empty results case
    if len(ref_comb_pvals) > 0:
        _, ref_fdr_pvals, _, _ = multitest.multipletests(ref_comb_pvals, alpha=alpha, method='fdr_bh')
        _, alt_fdr_pvals, _, _ = multitest.multipletests(alt_comb_pvals, alpha=alpha, method='fdr_bh')
    else:
        ref_fdr_pvals = []
        alt_fdr_pvals = []

    res = dict()
    for i in range(len(comb_names)):
        res[comb_names[i]] = ((ref_comb_pvals[i], alt_comb_pvals[i]), 
                              (ref_comb_es[i], alt_comb_es[i]),
                              (ref_fdr_pvals[i], alt_fdr_pvals[i]))
    
    # Structure result
    # We store the Union of all valid groups used across projects
    # Note: groups here is slightly ambiguous in merged context, we store list of sets
    res = {'groups': all_groups, 'snvs': res}
    res = {subname: res}
    
    # Saving logic
    # compressor = 'gz' # Default or derived from last file
    out_filename = f'{output_name}.comb.{compressor}'
    
    if os.path.isfile(out_filename):
        try:
            with openers[compressor](out_filename, 'rb') as f:
                r = dill.load(f)
                for t in r:
                    if t not in res:
                        res[t] = r[t]
        except Exception as e:
            logging.warning(f"Could not merge with existing file {out_filename}: {e}")

    if save_to_file:
        with openers[compressor](out_filename, 'wb') as f:
            dill.dump(res, f)
            
    return res