"""
Usage:
    calc_pval [options] (-I <file> ... | -f <file-list>) (-w <dir> -O <dir> -m <model>)
    calc_pval aggregate [options] (-I <file> ... | -f <file-list>) (-O <out>)

Arguments:
    <file>            Path to input file(s) in tsv format
    <file-list>       File with paths to input files
    <dir>             Directory name
    <out>             Output file path
    <int>             Positive integer
    <ext>             Extension, non-empty string
    <model>           String, one of 'line', 'window', 'NB_AS_Total'
    <dist>            String, one of 'NB', 'BetaNB'
    <method>          String, one of 'logit', 'fisher'

Required:
    -I <file>                               Input files
    -f <file-list>                          File with filenames of input files on each line
    -O <path>, --output <path>              Output director
    -w <dir>, --weights <dir>               Directory with fitted weights for models
    -m <model>, --model <model>             Model to calculate p-value with [default: NB_AS_Total]
    -d <dist>, --distribution <dist>        Underlying distribution to use in 'line' or 'window' models [default: BetaNB]
    -s <int>, --samples <int>               Minimal number of p-value samples required to run power-regression. Recommended
                                            values are 10-30. If 'inf', then power regression is disabled [default: inf].

Optional:
    -h, --help                              Show help
    --deprecated                            Use deprecated name for read count column (ADASTRA)
    --rescale-mode <mode>                   Mode for rescaling weights, one of (none, single, group) [default: none]
                                            obtained from SNPs at the respective position from all datasets
    --coverage-tr <int>                     Coverage threshold for aggregation step [default: 20]
    --method <method>                       Method for p-value aggregation [default: logit]
    --njobs <int>                           Number of jobs to use [default: 1]
    --gof-tr <float>                        Goodness of fit threshold [default: None]

Visualization
    -n, --no-fit                            Skip p-value calculation
    --visualize                             Perform visualization
    -e <ext>, --ext <ext>                   Extension to save figures with [default: svg]
"""

import os
from collections import namedtuple

from betanegbinfit import bridge_mixalime, ModelMixture
import pandas as pd
from scipy.stats import combine_pvalues
from statsmodels.stats import multitest
from scipy import stats as st
import multiprocessing as mp
import numpy as np
from negbin_fit.helpers import init_docopt, alleles, check_weights_path, get_inferred_mode_w, add_BAD_to_path, \
    merge_dfs, read_dfs, get_key, get_counts_column, get_p, get_pvalue_file_path, parse_files_list, parse_input, \
    available_models, available_bnb_models, available_dists, aggregation_methods
from schema import Schema, And, Const, Use, Or
from tqdm import tqdm


def calc_pval_for_model(row, row_weights, fit_params, model, gof_tr=0.1, allele_tr=5,
                        min_samples=np.inf, is_deprecated=False, rescale_mode='group'):
    if model in available_bnb_models:
        if gof_tr is None:
            gof_tr = np.inf
        params, models_dict = fit_params
        scaled_weights = {}
        BAD = row['BAD']
        for main_allele in alleles:
            w = params[main_allele][round(BAD, 2)]['params']['Estimate'].get('w{}'.format(
                row[get_counts_column(alleles[main_allele], is_deprecated=is_deprecated)]), 0.5)
            if rescale_mode == 'group':
                scaled_weights[main_allele] = modify_w_with_bayes_factor(w, row_weights[main_allele])
            elif rescale_mode == 'single':
                bayes_factor = np.exp(calc_log_bayes_factor(
                    k=row[get_counts_column(main_allele, is_deprecated=is_deprecated)],
                    m=row[get_counts_column(alleles[main_allele], is_deprecated=is_deprecated)],
                    BAD=BAD,
                    params=get_params_by_model(params, main_allele, BAD, model),
                    model=model
                ))
                scaled_weights[main_allele] = modify_w_with_bayes_factor(w, bayes_factor)
            else:
                scaled_weights[main_allele] = w
        pval, es = bridge_mixalime.calc_pvalue_and_es(
            ref_count=row[get_counts_column('ref', is_deprecated=is_deprecated)],
            alt_count=row[get_counts_column('alt', is_deprecated=is_deprecated)],
            params=params,
            w_ref=scaled_weights['ref'],
            w_alt=scaled_weights['alt'],
            m=models_dict[row['BAD']],
            gof_tr=gof_tr,
            concentration=250,
            min_samples=min_samples
            )

        if abs(pval[0]) > 1:
            print(row, pval, es, scaled_weights,
                  params['ref'][round(row['BAD'], 2)]['params']['Estimate'].get('w{}'.format(
                      row['{}_COUNTS'.format('alt'.upper())]), 0.5))
        return (*pval, *es)
    else:
        return calculate_pval_negbin(row, row_weights, fit_params, gof_tr, allele_tr)


def calculate_pval_negbin(row, row_weights, fit_params, gof_tr=0.1, allele_tr=5):
    """
    returns log2 ES
    """
    p_values = {}
    effect_sizes = {}
    for main_allele in alleles:
        gof = None
        BAD = row['BAD']
        p = 1 / (BAD + 1)
        r0, p0, w0, th0, gofs = get_neg_bin_params(fit_params, main_allele, BAD)
        k = row[get_counts_column(main_allele)]
        m = row[get_counts_column(alleles[main_allele])]
        if gofs is not None:
            gof = gofs.get(str(m), 0)

        if gof is None:
            p_values[main_allele] = np.nan
            effect_sizes[main_allele] = np.nan
            continue

        nb1 = st.nbinom(m + r0, 1 - (p * p0))
        geom1 = st.nbinom(m + 1, 1 - (p * th0))
        nb2 = st.nbinom(m + r0, 1 - ((1 - p) * p0))
        geom2 = st.nbinom(m + 1, 1 - ((1 - p) * th0))

        nb_w = get_inferred_mode_w(m, r0, p0, p)
        geom_w = get_inferred_mode_w(m, 1, th0, p)

        bayes_factor = row_weights[main_allele]

        nb_w = modify_w_with_bayes_factor(nb_w, bayes_factor)
        geom_w = modify_w_with_bayes_factor(geom_w, bayes_factor)

        cdf = get_dist_mixture(nb1.cdf, nb2.cdf, geom1.cdf, geom2.cdf, nb_w, geom_w, w0)
        pmf = get_dist_mixture(nb1.pmf, nb2.pmf, geom1.pmf, geom2.pmf, nb_w, geom_w, w0)

        p_values[main_allele] = (1 - cdf(k - 1)) / (1 - cdf(allele_tr - 1))
        if p_values[main_allele] != 1:
            sigma = cdf(allele_tr - 1)
            expectation = ((1 - w0) * ((1 - nb_w) * (m + r0) * p * p0 / (1 - p * p0)
                                       + nb_w * (m + r0) * (1 - p) * p0 / (1 - (1 - p) * p0))
                           + w0 * ((1 - geom_w) * (m + 1) * p * th0 / (1 - p * th0)
                                   + geom_w * (m + 1) * (1 - p) * th0 / (1 - (1 - p) * th0))
                           - sigma * sum(i * pmf(i) for i in range(1, allele_tr))) / \
                          (1 - sigma)
            effect_sizes[main_allele] = np.log2(k / expectation)
        else:
            effect_sizes[main_allele] = np.nan
    return p_values['ref'], p_values['alt'], effect_sizes['ref'], effect_sizes['alt']


def modify_w_with_bayes_factor(w, bf):
    if w != 0:
        if bf == np.inf:
            w = 0
        else:
            w = 1 / (1 + bf * (1 / w - 1))
    return w


def get_function_mixture(fun1, fun2, w):
    return lambda x: (1 - w) * fun1(x) + w * fun2(x)


def get_dist_mixture(nb1, nb2, geom1, geom2, nb_w, geom_w, w0):
    nb = get_function_mixture(nb1, nb2, nb_w)
    geom = get_function_mixture(geom1, geom2, geom_w)
    return get_function_mixture(nb, geom, w0)


def process_df(row, weights, fit_params, model, min_samples=np.inf, is_deprecated=False, gof_tr=None,
               rescale_mode='group'):
    p_ref, p_alt, es_ref, es_alt = calc_pval_for_model(row, weights.get(get_key(row, is_deprecated)), fit_params,
                                                       model, gof_tr=gof_tr, min_samples=min_samples,
                                                       is_deprecated=is_deprecated,
                                                       rescale_mode=rescale_mode)
    row[get_counts_column('ref', 'pval')] = p_ref
    row[get_counts_column('alt', 'pval')] = p_alt
    row[get_counts_column('ref', 'es')] = es_ref
    row[get_counts_column('alt', 'es')] = es_alt
    return row


def filter_df(merged_df, key):
    return merged_df[merged_df.key.values == key]


def get_neg_bin_params(fit_param, main_allele, BAD):
    try:
        fit_params = fit_param[BAD]
    except KeyError:
        print('No fit weights for BAD={} skipping ...'.format(BAD))
        return [None] * 5
    fit_params = fit_params[alleles[main_allele]]
    r0 = fit_params['r0']
    p0 = fit_params['p0']
    w0 = fit_params['w0']
    th0 = fit_params['th0']
    gofs = fit_params['point_gofs']
    return r0, p0, w0, th0, gofs


def get_pmf_for_dist(params, k, m, BAD, model):
    p = get_p(BAD)
    if model in available_bnb_models:
        logpdfs = list(map(lambda x: x[k] if x is not None and len(x) > k else None,
                           params['logpdf_notrunc']['modes'].get(m, (None, None))))
        return logpdfs
    else:
        r0, p0, w0, th0, _ = params
        nb1 = st.nbinom(m + r0, 1 - (p * p0))
        geom1 = st.nbinom(m + 1, 1 - (p * th0))
        nb2 = st.nbinom(m + r0, 1 - ((1 - p) * p0))
        geom2 = st.nbinom(m + 1, 1 - ((1 - p) * th0))
        pmf1 = lambda x: (1 - w0) * nb1.pmf(x) + w0 * geom1.pmf(x)
        pmf2 = lambda x: (1 - w0) * nb2.pmf(x) + w0 * geom2.pmf(x)
        pm1 = pmf1(k)
        pm2 = pmf2(k)
        if pm1 == 0:
            pm1 = nb1.logpmf(k)
        else:
            pm1 = np.log(pm1)
        if pm2 == 0:
            pm2 = nb2.logpmf(k)
        else:
            pm2 = np.log(pm2)
    return pm1, pm2


def get_params_by_model(fit_params, main_allele, BAD, model):
    if model in available_bnb_models:
        return fit_params[main_allele][round(BAD, 2)]
    else:
        return get_neg_bin_params(fit_params, main_allele, BAD)


def calc_log_bayes_factor(k, m, BAD, params, model):
    pm1, pm2 = get_pmf_for_dist(params, k, m, BAD, model)
    if pm1 is None or pm2 is None:
        add = 1
    else:
        add = pm1 - pm2  # log (1 - w) / w bayes factor
    return add


def calculate_posterior_weight_for_snp(filtered_df, model, fit_params, is_deprecated=False):
    result = {'ref': 0, 'alt': 0}
    cache = {}
    snp = filtered_df['key'].tolist()[0]
    BAD = filtered_df['BAD'].unique()
    assert len(BAD) == 1
    BAD = BAD[0]
    for main_allele in alleles:
        ks = filtered_df[get_counts_column(main_allele, is_deprecated=is_deprecated)].to_list()  # main_counts
        ms = filtered_df[get_counts_column(alleles[main_allele], is_deprecated=is_deprecated)].to_list()  # fixed_counts
        try:
            params = get_params_by_model(fit_params, main_allele, BAD, model)
        except KeyError:
            print(fit_params, main_allele, BAD, model)
            raise
        prod = np.float64(0)
        for k, m in zip(ks, ms):
            if (k, m, BAD, main_allele) in cache:
                add = cache[(k, m, BAD, main_allele)]
            else:
                add = calc_log_bayes_factor(k, m, BAD, params, model)
                if k + m <= 200:  # FIXME
                    cache[(k, m, BAD, main_allele)] = add
            prod += add
        with np.errstate(over='ignore'):
            prod = np.exp(prod)
        result[main_allele] = prod
    return result


def get_posterior_weights(merged_df, model, fit_params, is_deprecated=False):
    result = {}
    gb = merged_df.groupby('key')
    for filtered_df in tqdm([gb.get_group(x) for x in gb.groups]):
        result[filtered_df['key'].tolist()[0]] = calculate_posterior_weight_for_snp(
            filtered_df=filtered_df,
            model=model,
            fit_params=fit_params,
            is_deprecated=is_deprecated
        )
    return result


def start_process(dfs, merged_df, unique_BADs, out_path, fit_params, model, dist,
                  min_samples=np.inf, gof_tr=None, is_deprecated=False, rescale_mode=True):
    if rescale_mode == 'group':
        print('Calculating posterior weights...')
        weights = get_posterior_weights(merged_df, model, fit_params, is_deprecated=is_deprecated)
    else:
        weights = {}
    tqdm.pandas()
    if model in available_bnb_models:
        models_dict = {}
        for BAD in unique_BADs:
            # FIXME
            models_dict[BAD] = ModelMixture(bad=BAD, left=4, dist=dist)
        fit_params = fit_params, models_dict
    result = []
    for df_name, df in dfs:
        print('Calculating p-value for {}'.format(df_name))
        df = df.progress_apply(lambda x: process_df(x, weights, fit_params, model,
                                                    gof_tr=gof_tr,
                                                    min_samples=min_samples, is_deprecated=is_deprecated,
                                                    rescale_mode=rescale_mode), axis=1)
        df[[x for x in df.columns if x not in ('key', 'fname')]].to_csv(get_pvalue_file_path(out_path, df_name),
                                                                        sep='\t', index=False)
        result.append((df_name, df))
    return result


def check_fit_params_for_BADs(weights_path, BADs):
    result = {}
    for BAD in BADs:
        bad_weight_path = add_BAD_to_path(weights_path, BAD, create=False)
        result[BAD] = check_weights_path(bad_weight_path, False)[1]
    return result


def logit_combine_p_values(pvalues):
    pvalues = np.array([pvalue for pvalue in pvalues if 1 > pvalue > 0])
    if len(pvalues) == 0:
        return 1
    elif len(pvalues) == 1:
        return pvalues[0]

    statistic = -np.sum(np.log(pvalues)) + np.sum(np.log1p(-pvalues))
    k = len(pvalues)
    nu = np.int_(5 * k + 4)
    approx_factor = np.sqrt(np.int_(3) * nu / (np.int_(k) * np.square(np.pi) * (nu - np.int_(2))))
    pval = st.distributions.t.sf(statistic * approx_factor, nu)
    return pval


def aggregate_es(es_array, p_array):
    if len([x for x in es_array if not pd.isna(x)]) > 0:
        weights = [-1 * np.log10(x) for x in p_array if x != 1 and not pd.isna(x)]
        try:
            es_mean = np.round(np.average(es_array, weights=weights), 3)
        except TypeError:
            print(es_array, p_array)
            raise
        es_mostsig = es_array[int(np.argmax(weights))]
    else:
        es_mean = np.nan
        es_mostsig = np.nan
    return es_mean, es_mostsig


def get_es_list(filtered_df, allele):
    es_col = get_counts_column(allele, 'es')
    pval_col = get_counts_column(allele, 'pval')
    return [(row[es_col], row['fname']) for _, row in filtered_df.iterrows()
            if not pd.isna(row[es_col]) and row[pval_col] != 1]


def list_to_str(array):
    return '@'.join(array)


def combine_pvalues_with_method(p_array, method):
    if method == 'logit':
        return logit_combine_p_values(p_array)
    else:
        return combine_pvalues(pvalues=p_array, method=method)[1]


def aggregate_dfs(merged_df, unique_snps, method='logit'):
    result = []
    header = ['#CHROM', 'POS', 'ID',
              'REF', 'ALT', 'LOGITP_REF', 'ES_REF', 'LOGITP_ALT', 'ES_ALT']
    for snp in tqdm(unique_snps, unit='SNPs'):
        snp_result = snp.split('@')
        filtered_df = filter_df(merged_df, snp)
        conc_exps = {}
        for allele in alleles:
            conc_exps[allele] = {}
            fnames = set(filtered_df['fname'].unique())
            p_array = filtered_df[get_counts_column(allele, 'pval')].to_list()
            snp_result.append(combine_pvalues_with_method(p_array, method))
            es_fname_array = get_es_list(filtered_df, allele)
            sup_fnames = set([fname.strip() for es, fname in es_fname_array if es >= 0])
            conc_exps[allele]['sup'] = '@'.join(sup_fnames)
            conc_exps[allele]['non_sup'] = '@'.join(fnames - sup_fnames)
            es_mean, es_most_sig = aggregate_es([es for es, _ in es_fname_array], p_array)
            snp_result.append(es_mean)
        row_dict = dict(zip(header, snp_result))
        row_dict['MAX_COVER'] = filtered_df[[get_counts_column(allele) for allele in alleles]].sum(axis=1).max()
        row_dict['MEAN_BAD'] = filtered_df['BAD'].mean()
        for allele in alleles:
            for sup_type in 'sup', 'non_sup':
                row_dict[f'{sup_type.upper()}_{allele.upper()}_EXPS'] = conc_exps[allele][sup_type]

        result.append(row_dict)
    return pd.DataFrame(result)


def calc_fdr(aggr_df, max_cover_tr):
    mc_filter_array = np.array(aggr_df['MAX_COVER'] >= max_cover_tr)
    for allele in alleles:
        if sum(mc_filter_array) != 0:
            try:
                _, pval_arr, _, _ = multitest.multipletests(
                    aggr_df[mc_filter_array][f'LOGITP_{allele.upper()}'],
                    alpha=0.05, method='fdr_bh')
            except TypeError:
                print(aggr_df, aggr_df[mc_filter_array][f'LOGITP_{allele.upper()}'])
                raise
        else:
            pval_arr = []
        fdr_arr = np.empty(len(aggr_df.index), dtype=np.float128)
        fdr_arr[:] = np.nan
        fdr_arr[mc_filter_array] = pval_arr
        aggr_df[f"FDRP_BH_{allele.upper()}"] = fdr_arr
    return aggr_df


def main():
    schema = Schema({
        '-I': Or(
            Const(lambda x: x == []),
            And(
                Const(lambda x: sum(os.path.exists(y) for y in x), error='Input file(s) should exist'),
            )
        ),
        '-f': Or(
            Const(lambda x: x is None),
            Use(parse_files_list, error='Error while parsing file -f')
        ),
        '--weights': Or(
            Const(lambda x: x is None),
            And(
                Const(os.path.exists, error='Weights dir should exist'),
            )
        ),
        '--gof-tr': Or(
            And(
                Const(lambda x: x is None or x == 'None'),
                Use(lambda x: None)
            ),
            Use(float)),
        '--model': Const(lambda x: x in available_models,
                         error='Model not in ({})'.format(', '.join(available_models))),
        '--distribution': Const(lambda x: x in available_dists,
                                error='Distribution not in ({})'.format(', '.join(available_dists))),
        '--samples': Or(
            And(Use(int), Const(lambda x: x >= 0)),
            And(Use(str), Const(lambda x: x in ('inf',))),
            error='Samples must be a non negative integer or "inf" string',
        ),
        '--output': Or(
            And(
                Const(os.path.exists),
                Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
            ),
            And(
                Const(lambda x: not os.path.exists(x)),
                Const(lambda x: os.access(os.path.dirname(x) if os.path.dirname(x) != '' else '.', os.W_OK),
                      error='No write permissions')
            ),
        ),
        '--njobs': Use(lambda x: int(x)),
        '--coverage-tr': Use(lambda x: int(x)),
        '--ext': Const(lambda x: len(x) > 0),
        '--method': Const(lambda x: x in aggregation_methods,
                          error='Not in supported aggregation methods ({})'.format(
                              ', '.join(aggregation_methods))),
        '--rescale-mode': Const(lambda x: x in ('none', 'single', 'group')),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    is_deprecated = args['--deprecated']
    rescale_mode = args['--rescale-mode']
    dfs = parse_input(args['-I'], args['-f'], 0, is_deprecated=is_deprecated)
    out = args['--output']
    ext = args['--ext']
    model = args['--model']
    weights_dir = args['--weights']
    dist = args['--distribution']
    min_samples = args['--samples']
    if min_samples == 'inf':
        min_samples = np.inf
    unique_snps, unique_BADs, merged_df = merge_dfs([x[1] for x in dfs])
    if unique_snps is None:
        return
    if not args['aggregate']:
        if not os.path.exists(out):
            try:
                os.mkdir(out)
            except Exception:
                print(__doc__)
                print('Can not create output directory')
                raise
        if model in available_bnb_models:
            fit_params = bridge_mixalime.read_dist_from_folder(folder=weights_dir)
        else:
            try:
                fit_params = check_fit_params_for_BADs(weights_dir,
                                                       unique_BADs)
            except Exception as e:
                print(__doc__)
                print('Wrong format weights')
                raise e

        if not args['--no-fit']:
            result_dfs = start_process(
                merged_df=merged_df,
                out_path=out,
                dfs=dfs,
                unique_BADs=unique_BADs,
                fit_params=fit_params,
                model=args['--model'],
                gof_tr=args['--gof-tr'],
                dist=dist,
                min_samples=min_samples,
                is_deprecated=is_deprecated,
                rescale_mode=rescale_mode,
            )
        else:
            result_dfs = [pd.read_table(get_pvalue_file_path(out, df_name)) for df_name, df in dfs]
        if args['--visualize']:
            from calculate_pvalue.visualize import main as visualize
            visualize(dfs=result_dfs,
                      BADs=unique_BADs,
                      ext=ext,
                      out=out)

    else:
        if os.path.isdir(out):
            out = os.path.join(out, dfs[0][0] + '.tsv')
        aggregated_df = aggregate_dfs(merged_df, unique_snps, args['--method'])
        if aggregated_df.empty:
            raise AssertionError('No SNPs left after aggregation')
        fdr_df = calc_fdr(aggr_df=aggregated_df, max_cover_tr=args['--coverage-tr'])
        fdr_df['POS'] = fdr_df['POS'].astype(int)
        fdr_df['START'] = fdr_df['POS'] - 1
        fdr_df['END'] = fdr_df['POS']
        fdr_df[['#CHROM', 'START', 'END', 'ID', 'REF', 'ALT', 'MEAN_BAD', 'MAX_COVER',
                'LOGITP_REF', 'ES_REF', 'LOGITP_ALT', 'ES_ALT', 'SUP_REF_EXPS', 'NON_SUP_REF_EXPS',
                'SUP_ALT_EXPS', 'NON_SUP_ALT_EXPS',
                'FDRP_BH_REF', 'FDRP_BH_ALT'
                ]].to_csv(out, index=False, sep='\t')
