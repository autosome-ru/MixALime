# -*- coding: utf-8 -*-
from .utils import get_init_file, openers
from collections import defaultdict
import pandas as pd
import tarfile
import dill
import os


def export_counts(project, out: str, bad: float = None):
    if type(project) is str:
        file = get_init_file(project)
        compression = file.split('.')[-1]
        open = openers[compression]
        with open(file, 'rb') as f:
            counts_d = dill.load(f)['counts']
    else:
        counts_d = project['counts']
    if bad is None:
        for bad, counts in counts_d.items():
            subfolder = os.path.join(out, f'BAD{bad:.2f}')
            os.makedirs(subfolder, exist_ok=True)
            df = pd.DataFrame(counts, columns=['ref', 'alt', 'n'])
            df.to_csv(os.path.join(subfolder, 'counts.tsv'), sep='\t', index=None)
    else:
        df = pd.DataFrame(counts_d[bad], columns=['Ref', 'Alt', 'N'])
        df.to_csv(out, sep='\t', index=None)
    

def _export_params(fit, out: str, allele: str, bad: float):
    df = pd.DataFrame(fit[allele][bad]['params'])
    df.columns = ['Name', 'Estimate']
    df.to_csv(out, sep='\t', index=None)

def export_params(project, out: str, bad: float = None, allele: str = None):
    assert (bad is None) or (allele is not None), 'Both BAD and allele should be supplied.'
    if type(project) is str:
        file = get_init_file(project)
        compression = file.split('.')[-1]
        open = openers[compression]
        with open(f'{project}.fit.{compression}', 'rb') as f:
            fit = dill.load(f)
    else:
        fit = project
    if bad is None:
        for allele in fit:
            for bad in fit[allele]:
                subfolder = os.path.join(out, f'BAD{bad:.2f}')
                os.makedirs(subfolder, exist_ok=True)
                _export_params(fit, os.path.join(subfolder, f'param_{allele}.tsv'), 
                               allele, bad)
    else:
        _export_params(fit, out, allele, bad)
                

def _export_stats(params, out: str, allele: str, bad: float):
    d = defaultdict(list)
    for i, its in params[allele][bad]['stats'].items():
        d['slice'].append(i)
        for n, val in its.items():
            d[n].append(val)
    df = pd.DataFrame(d)
    df.to_csv(out, sep='\t', index=None)

def export_stats(project, out: str, bad: float = None, allele: str = None):
    assert (bad is None) or (allele is not None), 'Both BAD and allele should be supplied.'
    if type(project) is str:
        file = get_init_file(project)
        compression = file.split('.')[-1]
        open = openers[compression]
        with open(f'{project}.fit.{compression}', 'rb') as f:
            params = dill.load(f)
    else:
        params = project
    if bad is None:
        os.makedirs(out, exist_ok=True)
        for allele in params:
            for bad in params[allele]:
                subfolder = os.path.join(out, f'BAD{bad:.2f}')
                os.makedirs(subfolder, exist_ok=True)
                _export_stats(params, os.path.join(subfolder, f'stat_{allele}.tsv'), 
                               allele, bad)
    else:
        _export_stats(params, out, allele, bad)

def get_name(name: str):
    s = name.split('.')
    if len(s) == 1:
        return name
    for i, c in enumerate(s[::-1]):
        if c not in ('gz', 'vcf', 'bam', 'tsv', 'csv', 'zip', 'lzma', 'bz2', 'bed'):
            break
    return '.'.join(s[:-i])

def export_pvalues(project, out: str):
    if type(project) is str:
        file = get_init_file(project)
        compression = file.split('.')[-1]
        open = openers[compression]
        with open(file, 'rb') as snvs, open(f'{project}.test.{compression}', 'rb') as test:
            snvs = dill.load(snvs)
            test = dill.load(test)
    else:
        snvs, test = project
    scorefiles = snvs['scorefiles']
    snvs = snvs['snvs']  
    res = defaultdict(lambda: defaultdict(list))
    
    for (chr, pos), its in snvs.items():
        bad, name, ref, alt = its[0]
        end = pos + 1
        for filename_id, ref_count, alt_count in its[1:]:
            d = res[scorefiles[filename_id]]
            d['#chr'].append(chr)
            d['start'].append(pos)
            d['end'].append(end)
            d['bad'].append(bad)
            d['id'].append(name)
            d['ref'].append(ref)
            d['alt'].append(alt)
            d['ref_count'].append(ref_count)
            d['alt_count'].append(alt_count)
            (pval_ref, es_ref) = test['ref'][bad][(ref_count, alt_count)]
            (pval_alt, es_alt) = test['alt'][bad][(ref_count, alt_count)]
            d['ref_es'].append(es_ref)
            d['alt_es'].append(es_alt)
            d['ref_pval'].append(pval_ref)
            d['alt_pval'].append(pval_alt)
            if pval_ref < pval_alt:
                d['es'].append(es_ref)
                d['pval'].append(pval_ref)
            else:
                d['es'].append(es_alt)
                d['pval'].append(pval_alt)
    for file, d in res.items():
        folder, file = os.path.split(file)
        folder = os.path.join(out, folder)
        file = get_name(file) + '.pvalue.tsv'
        file = os.path.join(folder, file)
        os.makedirs(folder, exist_ok=True)
        pd.DataFrame(d).to_csv(file, sep='\t', index=None)

def export_combined_pvalues(project, out: str, rep_info=False, subname=None):
    if type(project) is str:
        file = get_init_file(project)
        compression = file.split('.')[-1]
        open = openers[compression]
        with open(file, 'rb') as snvs, open(f'{project}.test.{compression}', 'rb') as test, open(f'{project}.comb.{compression}', 'rb') as comb:
            snvs = dill.load(snvs)
            test = dill.load(test)
            comb = dill.load(comb)
    else:
        snvs, test, comb = project
    comb = comb[subname]
    scorefiles = snvs['scorefiles']
    snvs = snvs['snvs']  
    d = defaultdict(list)
    
    for (chr, pos), its in snvs.items():
        try:
            (pval_ref, pval_alt), (es_ref, es_alt), (fdr_ref, fdr_alt) = comb[(chr, pos)]
        except KeyError:
            continue
        bad, name, ref, alt = its[0]
        end = pos + 1
        ref_counts = list()
        alt_counts = list()
        scores_f = list()
        ref_pvals = list()
        alt_pvals = list()
        ref_eses = list()
        alt_eses = list()
        for filename_id, ref_count, alt_count in its[1:]:
            scores_f.append(scorefiles[filename_id])
            ref_counts.append(str(ref_count)); alt_counts.append(str(alt_count))
            (pval_r, es_r) = test['ref'][bad][(ref_count, alt_count)]
            (pval_a, es_a) = test['alt'][bad][(ref_count, alt_count)]
            ref_pvals.append(str(pval_r)); alt_pvals.append(str(pval_a))
            ref_eses.append(str(es_r)); alt_eses.append(str(es_a))
        ref_counts = ','.join(ref_counts);alt_counts = ','.join(alt_counts)
        ref_pvals = ','.join(ref_pvals); alt_pvals = ','.join(alt_pvals)
        ref_eses = ','.join(ref_eses); alt_eses = ','.join(alt_eses)
        n = len(scores_f)
        scores_f = ','.join(scores_f)
        d['#chr'].append(chr); d['start'].append(pos); d['end'].append(end); d['bad'].append(bad); d['id'].append(name)
        d['ref'].append(ref); d['alt'].append(alt); d['n_reps'].append(n);
        if rep_info:
            d['scorefiles'].append(scores_f)
            d['ref_counts'].append(ref_counts); d['alt_counts'].append(alt_counts); d['ref_es'].append(ref_eses); d['alt_es'].append(alt_eses)
            d['ref_pval'].append(ref_pvals); d['alt_pval'].append(alt_pvals); 
        d['ref_comb_es'].append(es_ref); d['alt_comb_es'].append(es_alt)
        d['ref_comb_pval'].append(pval_ref); d['alt_comb_pval'].append(pval_alt)
        d['ref_fdr_comb_pval'].append(fdr_ref); d['alt_fdr_comb_pval'].append(fdr_alt)
        if pval_ref < pval_alt:
            min_allele = 'ref'; fdr = fdr_ref; es = es_ref; pval = pval_ref
        else:
            min_allele = 'alt'; fdr = fdr_alt; es = es_alt; pval = pval_alt
        d['min_allele'].append(min_allele)
        d['comb_es'].append(es); d['comb_pval'].append(pval); d['fdr_comb_pval'].append(fdr)
        
    folder, _ = os.path.split(out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    pd.DataFrame(d).to_csv(out, sep='\t', index=None)

def export_difftests(project, out: str,  rep_info=False, subname=None):
    if type(project) is str:
        file = get_init_file(project)
        compression = file.split('.')[-1]
        open = openers[compression]
        with open(file, 'rb') as snvs,  open(f'{project}.difftest.{compression}', 'rb') as diff:
            diff = dill.load(diff)[subname]
    else:
        snvs, diff = project
        diff = diff[subname]
    tests = diff['tests']
    snvs_a, snvs_b = diff['snvs']
    chrom = list(); start = list(); end = list(); name = list(); bad = list(); ref = list(); alt = list()
    if rep_info:
        a_ref_counts = list(); b_ref_counts = list()
        a_alt_counts = list(); b_alt_counts = list()
    for ind in tests['ind']:
        b, n, r, a = snvs_a[ind][0]
        chrom.append(ind[0]); start.append(ind[1]); end.append(start[-1] + 1); bad.append(b); name.append(n);
        ref.append(r); alt.append(a)
        if rep_info:
            a_ref_count = list(); b_ref_count = list()
            a_alt_count = list(); b_alt_count = list()
            for _, r, a in snvs_a[ind][1:]:
                a_ref_count.append(str(r))
                a_alt_count.append(str(a))
            for _, r, a in snvs_b[ind][1:]:
                b_ref_count.append(str(r))
                b_alt_count.append(str(a))
            a_ref_counts.append(','.join(a_ref_count))
            b_ref_counts.append(','.join(b_ref_count))
            a_alt_counts.append(','.join(a_alt_count))
            b_alt_counts.append(','.join(b_alt_count))
    diff = tests.drop('ind', axis=1)
    if rep_info:
        df = pd.DataFrame({'#chr': chrom, 'start': start, 'end': end, 'bad': bad, 'id': name, 'ref': ref, 'alt': alt,
                           'a_ref_counts': a_ref_counts, 'a_alt_counts': a_alt_counts, 
                           'b_ref_counts': b_ref_counts, 'b_alt_counts': b_alt_counts})
    else:
        df = pd.DataFrame({'#chr': chrom, 'start': start, 'end': end, 'bad': bad, 'id': name, 'ref': ref, 'alt': alt })
    diff = pd.concat([df, diff], axis=1)
    t = diff['ref_pval'] < diff['alt_pval']
    mins = ['ref' if v else 'alt' for v in t]
    diff['min_allele'] = mins
    diff['pval'] = 1
    diff.loc[t, 'pval'] = diff.loc[t, 'ref_pval']
    diff.loc[~t, 'pval'] = diff.loc[~t, 'alt_pval']
    diff['fdr_pval'] = 1
    diff.loc[t, 'fdr_pval'] = diff.loc[t, 'ref_fdr_pval']
    diff.loc[~t, 'fdr_pval'] = diff.loc[~t, 'alt_fdr_pval']
    folder, _ = os.path.split(out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    diff.to_csv(out, sep='\t', index=None)



def export_all(name: str, out: str, rep_info: bool = None):
    file = get_init_file(name)
    compression = file.split('.')[-1]
    open = openers[compression]
    with open(file, 'rb') as init:
        init = dill.load(init)
    export_counts(init, out)
    try:
        with open(f'{name}.fit.{compression}', 'rb') as f:
            fit = dill.load(f)
    except FileNotFoundError:
        return
    export_params(fit, out)
    export_stats(fit, out)
    try:
        with open(f'{name}.difftest.{compression}', 'rb') as f:
            difftests = dill.load(f)
            subfolder = os.path.join(out, 'difftest')
            t = (init, difftests)
            for subname in difftests:
                export_difftests(t, os.path.join(subfolder, f'{subname}.tsv' if subname else 'difftests.tsv'), subname=subname,
                                 rep_info=rep_info)
    except FileNotFoundError:
        pass     
            
    try:
        with open(f'{name}.test.{compression}', 'rb') as f:
            raw_pvals = dill.load(f)
    except FileNotFoundError:
        return
    out = os.path.join(out, 'pvalues')
    export_pvalues((init, raw_pvals), os.path.join(out, 'raw'))
    try:
        with open(f'{name}.comb.{compression}', 'rb') as f:
            pvals = dill.load(f)
    except FileNotFoundError:
        return
    t = (init, raw_pvals, pvals)
    for subname in pvals:
        export_combined_pvalues(t, os.path.join(out, f'{subname}.tsv' if subname else 'pvals.tsv'), subname=subname, rep_info=rep_info)

def export_demo(path: str = str()):
    folder = os.path.join(os.path.split(os.path.realpath(__file__))[0], 'data')
    filename = os.path.join(folder, 'chip.tar.gz')
    with tarfile.open(filename, 'r:gz') as f:
        f.extractall(os.path.join(path, 'scorefiles'))
