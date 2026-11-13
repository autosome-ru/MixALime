# -*- coding: utf-8 -*-
from collections import defaultdict
from multiprocessing import cpu_count
from scipy import special
from scipy import stats as st
from statsmodels.stats import multitest
import hashlib
import logging
import numpy as np
import os
import pickle
import re
from .utils import get_init_file, load_pickle, openers, save_pickle, select_filenames, process_pool


# A process pool is started only if every process gets at least this many SNVs.
_MIN_SNVS_PER_JOB = 50000


def combine_p_values_logit(pvalues):
    """
    Combines p-values with the Mudholkar-George logit method.

    P-values equal to 1 are left out, as log1p(-p) is not finite there: a sample at 1
    does not weigh on the others, while one at 0.999999 does. The result is 1 if no p-value
    is left and the p-value itself if one is left. The tail of the statistic is taken from the t-distribution
    with 5k + 4 degrees of freedom, or from Fisher's chi-squared if the t tail underflows at a large k.
    """
    pvalues = np.array([x for x in pvalues if x < 1], dtype=float)
    k = len(pvalues)
    if k == 0:
        return 1.0
    if k == 1:
        return float(pvalues[0])
    with np.errstate(divide='ignore'):
        log_sum = np.log(pvalues).sum()
        statistic = -log_sum + np.log1p(-pvalues).sum()
    nu = 5 * k + 4
    approx_factor = np.sqrt(3 * nu / (k * np.square(np.pi) * (nu - 2)))
    pval = special.stdtr(nu, -statistic * approx_factor)
    if pval == 0:
        pval = st.distributions.chi2.sf(-2 * log_sum, 2 * k)
    return float(pval)


def combine_es(es, pvalues, uniform_weights=False):
    """
    Averages effect sizes with the weights -log10(p), or with equal weights if uniform_weights is set.

    A p-value of 0 is taken as the smallest positive float, so its sample is kept with the largest weight.
    Samples whose p-value is not a number are left out. A sample of weight 0 adds nothing, even if its effect size is infinite.
    """
    pvalues = np.array(list(map(float, pvalues)))
    es = np.array(es)
    weights = -np.log10(np.maximum(pvalues, np.finfo(float).tiny))
    inds = np.isfinite(weights)
    if not inds.any():
        return float(np.mean(es))
    weights = weights[inds]
    if uniform_weights:
        weights[:] = 1.0
    es = es[inds]
    s = weights.sum()
    weights = np.full(len(weights), 1 / len(weights)) if s == 0 else weights / s
    keep = weights > 0
    return float(np.sum(weights[keep] * es[keep]))


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


def _combine_snvs(items, stats, min_cover, groups, uniform_weights):
    """
    Combines the samples of every SNV in items, each item being (key, metadata, records of each project).

    Returns (key, ref p-value, alt p-value, ref effect size, alt effect size) for the SNVs with a
    sample in the groups whose ref + alt reaches the minimal cover of its BAD in the project.
    """
    ref_tests = [t['ref'] for t in stats]
    alt_tests = [t['alt'] for t in stats]
    res = list()
    for key, _, per_project in items:
        # A BAD that the fit skipped has no test results and no minimal cover.
        samples = [(p, ref, alt, bad) for p, records in enumerate(per_project)
                   for file, ref, alt, bad in records if file in groups[p] and bad in min_cover[p]]
        if not any(ref + alt >= min_cover[p][bad] for p, ref, alt, bad in samples):
            continue
        ref_pvals, ref_es = zip(*[ref_tests[p][bad][ref, alt] for p, ref, alt, bad in samples])
        alt_pvals, alt_es = zip(*[alt_tests[p][bad][ref, alt] for p, ref, alt, bad in samples])
        res.append((key, combine_p_values_logit(ref_pvals), combine_p_values_logit(alt_pvals),
                    combine_es(ref_es, ref_pvals, uniform_weights=uniform_weights),
                    combine_es(alt_es, alt_pvals, uniform_weights=uniform_weights)))
    return res


def _combine_chunk(task):
    items, context = task
    return _combine_snvs(items, *pickle.loads(context))


def _run(items, context, n_jobs):
    """
    Applies _combine_snvs to items and returns the rows in the order of items.

    A pool of n_jobs processes is used if there are enough SNVs, and then the rows are the same as without it.
    The pickled context is sent with the chunks, as passing it to the workers at their start is much slower.
    """
    n_jobs = min(n_jobs, len(items) // _MIN_SNVS_PER_JOB)
    if n_jobs <= 1:
        return _combine_snvs(items, *context)
    context = pickle.dumps(context, protocol=4)
    size = -(-len(items) // n_jobs)
    tasks = [(items[i:i + size], context) for i in range(0, len(items), size)]
    with process_pool(n_jobs) as p:
        return [row for part in p.map(_combine_chunk, tasks, chunksize=1) for row in part]


def _n_jobs(n_jobs) -> int:
    """The number of processes for n_jobs, where -1 stands for all cores. It is at least 1."""
    return max(1, cpu_count() if n_jobs == -1 else int(n_jobs or 1))


def _load_project(name: str):
    """Returns the SNVs, the scorefiles, the p-values and the compression of the project name."""
    filename = get_init_file(name)
    compressor = filename.split('.')[-1]
    init = load_pickle(filename)
    return init['snvs'], init['scorefiles'], load_pickle(f'{name}.test.{compressor}'), compressor


def project_signature(name: str, scorefiles: list, compressor: str) -> dict:
    """
    The base name of the project name with the SHA-256 digests of its list of scorefiles and of its test file.

    A result combined across projects keeps the signatures of its projects, so that an export can tell projects
    that are swapped or tested again from those that were combined.
    """
    digest = hashlib.sha256()
    with open(f'{name}.test.{compressor}', 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return {'name': os.path.basename(name), 'scorefiles': hashlib.sha256(repr(list(scorefiles)).encode()).hexdigest(),
            'test': digest.hexdigest()}


def _min_cover(stats, min_cnt_sum, adaptive_min_cover, adaptive_es, adaptive_pval):
    """Returns the minimal ref + alt of a sample for each BAD, and the adaptive coverages (None if they are not used)."""
    if not adaptive_min_cover:
        return {bad: min_cnt_sum for bad in stats['ref']}, None
    coverage = estimate_min_coverage(stats, es_cutoff=adaptive_es, pval_cutoff=adaptive_pval)
    return {bad: min(coverage['ref'][bad], coverage['alt'][bad]) for bad in coverage['ref']}, coverage


def _select_groups(patterns, scorefiles) -> set:
    """The indices of the scorefiles that agree with the patterns."""
    indices = {file: i for i, file in enumerate(scorefiles)}
    return {indices[file] for file in select_filenames([patterns] if isinstance(patterns, str) else patterns, scorefiles)}


def _filter_snvs(items, filter_id, filter_chr):
    """Selects the items whose ID and chromosome match the compiled regexes. An SNV without an ID does not match filter_id."""
    if filter_id is not None:
        items = [x for x in items if x[1][0] and filter_id.match(str(x[1][0]))]
    if filter_chr is not None:
        items = [x for x in items if filter_chr.match(str(x[0][0]))]
    return items


def _nonzero(pvalues) -> np.ndarray:
    """The p-values with the zeros replaced by the smallest nonzero one, or by the smallest positive float if all are zeros."""
    pvalues = np.array(pvalues, dtype=float)
    zeros = pvalues == 0.0
    if zeros.any():
        pvalues[zeros] = np.finfo(float).tiny if zeros.all() else pvalues[~zeros].min()
    return pvalues


def _finish(rows, groups, alpha, subname, filename, save_to_file, projects=None):
    """
    Corrects the combined p-values for multiple testing and stores them under subname in filename.

    The results kept in filename under other subnames are retained. The file is replaced
    atomically, so a failed write leaves the previous file as it was. The result keeps the
    signatures of the projects if they are given.
    """
    keys, ref_pvals, alt_pvals, ref_es, alt_es = zip(*rows) if rows else ((),) * 5
    ref_pvals, alt_pvals = _nonzero(ref_pvals), _nonzero(alt_pvals)
    if rows:
        ref_fdr = multitest.multipletests(ref_pvals, alpha=alpha, method='fdr_bh')[1]
        alt_fdr = multitest.multipletests(alt_pvals, alpha=alpha, method='fdr_bh')[1]
    else:
        logging.warning('No SNVs passed the filters, the result is empty.')
        ref_fdr = alt_fdr = ref_pvals
    snvs = dict(zip(keys, zip(zip(ref_pvals.tolist(), alt_pvals.tolist()), zip(ref_es, alt_es),
                              zip(ref_fdr.tolist(), alt_fdr.tolist()))))
    res = {subname: {'groups': groups, 'snvs': snvs}}
    if projects is not None:
        res[subname]['projects'] = projects
    if os.path.isfile(filename):
        try:
            for key, value in load_pickle(filename).items():
                res.setdefault(key, value)
        except Exception as e:
            logging.warning(f'The combined p-values in {filename} cannot be read ({e}) and are replaced.')
    if save_to_file:
        save_pickle(res, filename)
    return res


def combine(name: str, group_files=None, alpha=0.05, min_cnt_sum=20, adaptive_min_cover=False, adaptive_es=1.0, adaptive_pval=0.05,
            uniform_weights=False, filter_id=None, filter_chr=None, subname=None, n_jobs=1, save_to_file=True):
    """
    Combines the p-values of the samples of each SNV of the project name and corrects them for multiple testing.

    An SNV is kept if at least one of its samples among group_files (all samples if None) has ref + alt
    of at least min_cnt_sum; then all its samples among group_files are combined. filter_id and filter_chr
    are regexes for the ID and the chromosome. The result goes to name.comb.* under subname, and is
    returned with the adaptive coverages (None unless adaptive_min_cover is set).
    """
    n_jobs = _n_jobs(n_jobs)
    filter_id, filter_chr = (None if x is None else re.compile(x) for x in (filter_id, filter_chr))
    snvs, scorefiles, stats, compressor = _load_project(name)
    if group_files is None:
        groups = set(range(len(scorefiles)))
    else:
        groups = _select_groups(group_files, scorefiles)
        if not groups:
            raise SyntaxError('No files found for given pattern(s).')
    min_cover, adaptive_coverage = _min_cover(stats, min_cnt_sum, adaptive_min_cover, adaptive_es, adaptive_pval)
    items = _filter_snvs([(key, record[0], (record[1:],)) for key, record in snvs.items()], filter_id, filter_chr)
    del snvs
    rows = _run(items, ([stats], [min_cover], [groups], uniform_weights), n_jobs)
    res = _finish(rows, groups, alpha, subname, f'{name}.comb.{compressor}', save_to_file)
    return res, adaptive_coverage


def polymbine(output_name: str, names: list[str], group_files=None, alpha=0.05, min_cnt_sum=20,
              adaptive_min_cover=False, adaptive_es=1.0, adaptive_pval=0.05,
              uniform_weights=False, filter_id=None, filter_chr=None, subname=None,
              n_jobs=1, save_to_file=True):
    """
    Combines the p-values of the samples of the projects names as if they were replicates of one project.

    The SNV and sample rules are those of combine, with the samples of all projects taken together. If group_files
    has one pattern per project, the pattern i selects the samples of project i, otherwise every pattern is applied
    to every project. A project with no file selected adds no samples, and it is an error if no project has one.
    output_name is no project. The result goes to output_name.comb.*, in the compression of an existing such file or
    else of the first project. The adaptive coverages returned are a list with those of each project, or None.
    """
    if not names:
        raise ValueError('No projects are given.')
    try:
        get_init_file(output_name)
    except FileNotFoundError:
        pass
    else:
        raise ValueError(f'{output_name} is a project and its combined p-values would be mixed with those across projects. '
                         'Choose another output name.')
    if isinstance(group_files, str):
        group_files = [group_files]
    n_jobs = _n_jobs(n_jobs)
    filter_id, filter_chr = (None if x is None else re.compile(x) for x in (filter_id, filter_chr))
    merged = dict()
    stats_list, min_covers, groups_list, compressors, signatures, coverages = list(), list(), list(), list(), list(), list()
    for p, name in enumerate(names):
        snvs, scorefiles, stats, compressor = _load_project(name)
        if group_files:
            groups = _select_groups([group_files[p]] if len(group_files) == len(names) else group_files, scorefiles)
            if not groups:
                logging.warning(f'No files of the project {name} agree with the group patterns, it adds no samples.')
        else:
            groups = set(range(len(scorefiles)))
        min_cover, coverage = _min_cover(stats, min_cnt_sum, adaptive_min_cover, adaptive_es, adaptive_pval)
        for key, record in snvs.items():
            if key not in merged:
                merged[key] = (key, record[0], [()] * len(names))
            merged[key][2][p] = record[1:]
        stats_list.append(stats)
        min_covers.append(min_cover)
        groups_list.append(groups)
        compressors.append(compressor)
        signatures.append(project_signature(name, scorefiles, compressor))
        coverages.append(coverage)
        del snvs
    if group_files and not any(groups_list):
        raise SyntaxError('No files found for given pattern(s).')
    items = _filter_snvs(list(merged.values()), filter_id, filter_chr)
    del merged
    rows = _run(items, (stats_list, min_covers, groups_list, uniform_weights), n_jobs)
    existing = [f'{output_name}.comb.{c}' for c in openers if os.path.isfile(f'{output_name}.comb.{c}')]
    filename = max(existing, key=os.path.getmtime) if existing else f'{output_name}.comb.{compressors[0]}'
    res = _finish(rows, groups_list, alpha, subname, filename, save_to_file, signatures)
    return res, coverages if adaptive_min_cover else None
