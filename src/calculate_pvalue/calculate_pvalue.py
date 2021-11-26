"""
Usage:
    calc_pval (-I <file> ...) [--visualize] [-n | --no-fit] (-w <dir> | --weights <dir>) [-O <dir> |--output <dir>]
    calc_pval -h | --help
    calc_pval aggregate (-O <out> |--output <out>) (-I <file>...)

Arguments:
    <file>            Path to input file(s) in tsv format
    <dir>             Directory name
    <out>             Output file path

Options:
    -h, --help                              Show help.
    -n, --no-fit                            Skip p-value calculation
    -I <file>                               Input files
    -O <path>, --output <path>              Output directory. [default: ./]
    -w <dir>, --weights <dir>               Directory with fitted weights
"""

import os

import pandas as pd
from statsmodels.stats import multitest
from scipy import stats as st
import numpy as np
from negbin_fit.helpers import init_docopt, alleles, check_weights_path, get_inferred_mode_w, add_BAD_to_path, \
    merge_dfs, read_dfs, get_key, get_counts_column, get_p
from schema import Schema, And, Const, Use, Or
from tqdm import tqdm


def calculate_pval(row, row_weights, fit_params, gof_tr=0.1, allele_tr=5):
    """
    returns log2 ES
    """
    p_values = {}
    effect_sizes = {}
    for main_allele in alleles:
        gof = None
        BAD = row['BAD']
        p = 1 / (BAD + 1)
        r0, p0, w0, th0, gofs = get_params(fit_params, main_allele, BAD, row['key'])
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


def process_df(row, weights, fit_params):
    p_ref, p_alt, es_ref, es_alt = calculate_pval(row, weights[get_key(row)], fit_params)
    row[get_counts_column('ref', 'pval')] = p_ref
    row[get_counts_column('alt', 'pval')] = p_alt
    row[get_counts_column('ref', 'es')] = es_ref
    row[get_counts_column('alt', 'es')] = es_alt
    return row


def filter_df(merged_df, key):
    return merged_df[merged_df['key'] == key]


def get_params(fit_param, main_allele, BAD, err_id):
    try:
        fit_params = fit_param[BAD]
    except KeyError:
        print('No fit weights for BAD={}, {} skipping ...'.format(BAD, err_id))
        return [None] * 5
    fit_params = fit_params[alleles[main_allele]]
    r0 = fit_params['r0']
    p0 = fit_params['p0']
    w0 = fit_params['w0']
    th0 = fit_params['th0']
    gofs = fit_params['point_gofs']
    return r0, p0, w0, th0, gofs


def get_posterior_weights(merged_df, unique_snps, fit_params):
    result = {}
    cache = {}
    for snp in tqdm(unique_snps):
        result[snp] = {'ref': 0, 'alt': 0}
        filtered_df = filter_df(merged_df, snp)
        BAD = filtered_df['BAD'].unique()
        assert len(BAD) == 1
        BAD = BAD[0]
        p = get_p(BAD)
        for main_allele in alleles:
            ks = filtered_df[get_counts_column(main_allele)].to_list()  # main_counts
            ms = filtered_df[get_counts_column(alleles[main_allele])].to_list()  # fixed_counts
            r0, p0, w0, th0, _ = get_params(fit_params, main_allele, BAD, snp)
            prod = np.float64(0)
            for k, m in zip(ks, ms):
                if (k, m, BAD, main_allele) in cache:
                    add = cache[(k, m, BAD, main_allele)]
                else:
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
                    add = pm1 - pm2  # log (1 - w) / w bayes factor
                    if k + m <= 200:
                        cache[(k, m, BAD, main_allele)] = add
                prod += add
            with np.errstate(over='ignore'):
                prod = np.exp(prod)
            result[snp][main_allele] = prod

    return result


def start_process(dfs, merged_df, unique_snps, out_path, fit_params):
    print('Calculating posterior weights...')
    weights = get_posterior_weights(merged_df, unique_snps, fit_params)
    tqdm.pandas()
    for df_name, df in dfs:
        print('Calculating p-value for {}'.format(df_name))
        df = df.progress_apply(lambda x: process_df(x, weights, fit_params), axis=1)
        df[[x for x in df.columns if x != 'key']].to_csv(os.path.join(out_path, df_name + '.pvalue_table'),
                                                         sep='\t', index=False)


def check_fit_params_for_BADs(weights_path, BADs):
    result = {}
    for BAD in BADs:
        bad_weight_path = add_BAD_to_path(weights_path, BAD, create=False)
        result[BAD] = check_weights_path(bad_weight_path, True)[1]
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
        weights = [-1 * np.log10(x) for x in p_array if x != 1]
        es_mean = np.round(np.average(es_array, weights=weights), 3)
        es_mostsig = es_array[int(np.argmax(weights))]
    else:
        es_mean = np.nan
        es_mostsig = np.nan
    return es_mean, es_mostsig


def aggregate_dfs(merged_df, unique_snps):
    result = []
    header = ['#CHROM', 'POS', 'ID',
              'ALT', 'REF', 'MAXC_REF', 'LOGITP_REF', 'ES_REF', 'MAXC_ALT', 'LOGITP_ALT', 'ES_ALT']
    for snp in unique_snps:
        snp_result = snp.split(';')
        filtered_df = filter_df(merged_df, snp)
        for allele in alleles:
            max_c = max(filtered_df[get_counts_column(allele)].to_list())
            snp_result.append(max_c)
            p_array = filtered_df[get_counts_column(allele, 'pval')].to_list()
            snp_result.append(logit_combine_p_values(p_array))
            es_array = filtered_df[get_counts_column(allele, 'es')].to_list()
            es_mean, es_most_sig = aggregate_es([x for x in es_array if not pd.isna(x)], p_array)
            snp_result.append(es_mean)
        result.append(dict(zip(header, snp_result)))
    return pd.DataFrame(result)


def main():
    schema = Schema({
        '-I': And(
            Const(lambda x: sum(os.path.exists(y) for y in x), error='Input file(s) should exist'),
            Use(read_dfs, error='Wrong format stats file')
        ),
        '--weights': Or(
            Const(lambda x: x is None),
            And(
                Const(os.path.exists, error='Weights dir should exist'),
            )
        ),
        '--output': Or(
            And(
                Const(os.path.exists),
                Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
            ),
            And(
                Const(lambda x: not os.path.exists(x)),
                Const(lambda x: os.access(os.path.dirname(x) if os.path.dirname(x) != '' else '.', os.W_OK),
                      error='No write permissions'),
                Const(lambda x: os.mkdir(x) or True, error='Can not create output directory')
            ),
        ),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    dfs = args['-I']
    out = args['--output']
    ext = 'svg'
    unique_snps, unique_BADs, merged_df = merge_dfs([x[1] for x in dfs])
    if not args['aggregate']:
        try:
            weights = check_fit_params_for_BADs(args['--weights'],
                                                unique_BADs)
        except Exception:
            print(__doc__)
            exit('Wrong format weights')
            raise

        if not args['--no-fit']:
            start_process(
                merged_df=merged_df,
                out_path=out,
                dfs=dfs,
                unique_snps=unique_snps,
                fit_params=weights
            )
        if args['--visualize']:
            from calculate_pvalue.visualize import main as visualize
            for df in dfs:
                visualize(df=df,
                          BADs=unique_BADs,
                          ext=ext,
                          out=out)
    else:
        if os.path.isdir(out):
            out = os.path.join(out, dfs[0][0])
        aggregated_df = aggregate_dfs(merged_df, unique_snps)
        if aggregated_df.empty:
            raise AssertionError('No SNPs left after aggregation')

        mc_filter_array = np.array(aggregated_df[['MAXC_REF', 'MAXC_ALT']].max(axis=1) >= 20)
        if sum(mc_filter_array) != 0:
            bool_ar_ref, p_val_ref, _, _ = multitest.multipletests(
                aggregated_df[mc_filter_array]["LOGITP_REF"],
                alpha=0.05, method='fdr_bh')
            bool_ar_alt, p_val_alt, _, _ = multitest.multipletests(
                aggregated_df[mc_filter_array]["LOGITP_ALT"],
                alpha=0.05, method='fdr_bh')
        else:
            p_val_ref = []
            p_val_alt = []

        fdr_by_ref = np.array(['NaN'] * len(aggregated_df.index), dtype=np.float64)
        fdr_by_ref[mc_filter_array] = p_val_ref
        aggregated_df["FDRP_BH_REF"] = fdr_by_ref

        fdr_by_alt = np.array(['NaN'] * len(aggregated_df.index), dtype=np.float64)
        fdr_by_alt[mc_filter_array] = p_val_alt
        aggregated_df["FDRP_BH_ALT"] = fdr_by_alt
        aggregated_df.to_csv(out, index=False, sep='\t')
