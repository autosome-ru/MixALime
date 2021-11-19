"""
Usage:
    calc_pval (-w <dir> | --weights <dir>) [-O <dir> |--output <dir>] <file> ...
    calc_pval -h | --help

Arguments:
    <file>            Path to input file(s) in tsv format
    <dir>             Directory name

Options:
    -h, --help                              Show help.
    -O <path>, --output <path>              Output directory. [default: ./]
    -w <dir>, --weights <dir>               Directory with fitted weights
"""

import os
from scipy import stats as st
import numpy as np
import re
from negbin_fit.helpers import init_docopt, alleles, check_weights_path, get_inferred_mode_w, add_BAD_to_path, \
    merge_dfs, read_dfs, get_key, get_counts_column
from schema import Schema, And, Const, Use


def calculate_pval(row, row_weights, fit_params, gof_tr=0.1, allele_tr=5):
    """
    returns log2 ES
    """
    p_values = {}
    effect_sizes = {}
    for main_allele in alleles:
        gof = 0
        BAD = row['BAD']
        p = 1 / (BAD + 1)
        r0, p0, w0, th0, gofs = get_params(fit_params, main_allele, BAD, row['key'])
        k = row[get_counts_column(main_allele)]
        m = row[get_counts_column(alleles[main_allele])]
        if gofs is not None:
            gof = gofs.get(str(m), 0)

        if gof == 0 or gof > gof_tr:
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
                           - sigma * sum(i * pmf(i) for i in range(allele_tr))) /\
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
    row['pval_ref'] = p_ref
    row['pval_alt'] = p_alt
    row['es_ref'] = es_ref
    row['es_alt'] = es_alt
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
    for snp in unique_snps:
        result[snp] = {'ref': 0, 'alt': 0}
        filtered_df = filter_df(merged_df, snp)
        for main_allele in alleles:
            ks = filtered_df[get_counts_column(main_allele)].to_list()  # main_counts
            ms = filtered_df[get_counts_column(alleles[main_allele])].to_list()  # fixed_counts
            BAD = filtered_df['BAD'].unique()
            assert len(BAD) == 1
            BAD = BAD[0]
            p = 1 / (BAD + 1)
            r0, p0, w0, th0, _ = get_params(fit_params, main_allele, BAD, snp)
            prod = np.float64(1)
            for k, m in zip(ks, ms):
                nb1 = st.nbinom(m + r0, 1 - (p*p0))
                geom1 = st.nbinom(m + 1, 1 - (p*th0))
                nb2 = st.nbinom(m + r0, 1 - ((1 - p)*p0))
                geom2 = st.nbinom(m + 1, 1 - ((1 - p)*th0))
                pmf1 = lambda x: (1 - w0) * nb1.pmf(x) + w0 * geom1.pmf(x)
                pmf2 = lambda x: (1 - w0) * nb2.pmf(x) + w0 * geom2.pmf(x)
                prod *= pmf1(k) / pmf2(k)  # (1 - w) / w bayes factor
            result[snp][main_allele] = prod

    return result


def start_process(dfs, merged_df, unique_snps, out_path, fit_params):
    weights = get_posterior_weights(merged_df, unique_snps, fit_params)
    for df_name, df in dfs:
        df = df.apply(lambda x: process_df(x, weights, fit_params), axis=1)
        df.to_csv(os.path.join(out_path, df_name + '.pvalue_table'),
                  sep='\t', index=False)


def check_fit_params_for_BADs(weights_path, BADs):
    result = {}
    for BAD in BADs:
        bad_weight_path = add_BAD_to_path(weights_path, BAD)
        result[BAD] = check_weights_path(bad_weight_path, True)[1]
    return result


def check_states(string):
    if not string:
        return False
    string = string.strip().split(',')
    ret_val = list(map(convert_frac_to_float, string))
    if not all(ret_val):
        return False
    else:
        return ret_val


def convert_frac_to_float(string):
    if re.match(r"^[1-9]+[0-9]*/[1-9]+[0-9]*$", string):
        num, denom = string.split('/')
        if int(denom) <= 0:
            return False
        else:
            value = int(num) / int(denom)
    elif re.match(r"^[1-9]+[0-9]*\.[1-9]+[0-9]*$", string):
        try:
            value = float(string)
        except ValueError:
            return False
    elif re.match(r"^[1-9]+[0-9]*$", string):
        try:
            value = int(string)
        except ValueError:
            return False
    else:
        return False
    if value >= 1:
        return value
    else:
        return False


def main():
    schema = Schema({
        '<file>': And(
            Const(lambda x: sum(os.path.exists(y) for y in x), error='Input file(s) should exist'),
            Use(read_dfs, error='Wrong format stats file')
        ),
        '--weights': And(
            Const(os.path.exists, error='Weights dir should exist'),
        ),
        '--output': And(
            Const(os.path.exists),
            Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
        ),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    dfs = args['<file>']
    unique_snps, unique_BADs, merged_df = merge_dfs([x[1] for x in dfs])
    try:
        weights = check_fit_params_for_BADs(args['--weights'],
                                            unique_BADs)
    except Exception:
        print(__doc__)
        exit('Wrong format weights')
        raise
    start_process(
        merged_df=merged_df,
        out_path=args['--output'],
        dfs=dfs,
        unique_snps=unique_snps,
        fit_params=weights
    )
