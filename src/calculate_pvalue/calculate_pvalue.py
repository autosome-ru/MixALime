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
    <model>           String, one of (BetaNB, NB_AS_Total)

Required:
    -I <file>                               Input files
    -f <file-list>                          File with filenames of input files on each line
    -O <path>, --output <path>              Output director
    -w <dir>, --weights <dir>               Directory with fitted weights for models
    -m <model>, --model <model>             Model to calculate p-value with [default: NB_AS_Total]

Optional:
    -h, --help                              Show help
    --coverage-tr <int>                     Coverage threshold for aggregation step [default: 20]

Visualization
    -n, --no-fit                            Skip p-value calculation
    --visualize                             Perform visualization
    -e <ext>, --ext <ext>                   Extension to save figures with [default: svg]
"""

import os

from betanegbinfit import bridge_mixalime, ModelMixture
import pandas as pd
from statsmodels.stats import multitest
from scipy import stats as st
import numpy as np
from negbin_fit.helpers import init_docopt, alleles, check_weights_path, get_inferred_mode_w, add_BAD_to_path, \
    merge_dfs, read_dfs, get_key, get_counts_column, get_p, get_pvalue_file_path, parse_files_list, parse_input, \
    available_models
from schema import Schema, And, Const, Use, Or
from tqdm import tqdm


def calc_pval_for_model(row, row_weights, fit_params, model, gof_tr=0.1, allele_tr=5):
    if model == 'BetaNB':
        params, models_dict = fit_params
        pval, es = bridge_mixalime.calc_pvalue_and_es(ref_count=row['REF_COUNTS'],
                                                      alt_count=row['ALT_COUNTS'],
                                                      params=params,
                                                      w_ref=1,
                                                      w_alt=1,
                                                      m=models_dict[row['BAD']]
                                                      )
        return *pval, *es
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


def process_df(row, weights, fit_params, model):
    p_ref, p_alt, es_ref, es_alt = calc_pval_for_model(row, weights.get(get_key(row)), fit_params, model)
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


def start_process(dfs, merged_df, unique_snps, out_path, fit_params, model):
    if model == 'BetaNB':
        weights = {}
    else:
        print('Calculating posterior weights...')
        weights = get_posterior_weights(merged_df, unique_snps, fit_params)

    tqdm.pandas()
    result = []
    for df_name, df in dfs:
        print('Calculating p-value for {}'.format(df_name))
        df = df.progress_apply(lambda x: process_df(x, weights, fit_params, model), axis=1)
        df[[x for x in df.columns if x not in ('key', 'fname')]].to_csv(get_pvalue_file_path(out_path, df_name),
                                                                        sep='\t', index=False)
        result.append((df_name, df))
    return result


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


def get_es_list(filtered_df, allele):
    es_col = get_counts_column(allele, 'es')
    pval_col = get_counts_column(allele, 'pval')
    return [(row[es_col], row['fname']) for _, row in filtered_df.iterrows()
            if not pd.isna(row[es_col]) and row[pval_col] != 1]


def list_to_str(array):
    return '@'.join(array)


def aggregate_dfs(merged_df, unique_snps):
    result = []
    header = ['#CHROM', 'POS', 'ID',
              'ALT', 'REF', 'LOGITP_REF', 'ES_REF', 'LOGITP_ALT', 'ES_ALT']
    for snp in tqdm(unique_snps, unit='SNPs'):
        snp_result = snp.split('@')
        filtered_df = filter_df(merged_df, snp)
        conc_exps = {}
        for allele in alleles:
            p_array = filtered_df[get_counts_column(allele, 'pval')].to_list()
            snp_result.append(logit_combine_p_values(p_array))
            es_fname_array = get_es_list(filtered_df, allele)
            conc_exps[allele] = list_to_str(set([fname.strip() for es, fname in es_fname_array if es >= 0]))
            es_mean, es_most_sig = aggregate_es([es for es, _ in es_fname_array], p_array)
            snp_result.append(es_mean)
        row_dict = dict(zip(header, snp_result))
        row_dict['MAX_COVER'] = filtered_df[[get_counts_column(allele) for allele in alleles]].sum(axis=1).max()

        for allele in alleles:
            row_dict['{}_EXPS'.format(allele.upper())] = conc_exps[allele]

        result.append(row_dict)
    return pd.DataFrame(result)


def main():
    schema = Schema({
        '-I': Or(
            Const(lambda x: x == []),
            And(
                Const(lambda x: sum(os.path.exists(y) for y in x), error='Input file(s) should exist'),
                Use(read_dfs, error='Wrong format stats file')
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
        '--model': Const(lambda x: x in available_models,
                         error='Model not in ({})'.format(', '.join(available_models))),
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
        '--coverage-tr': Use(lambda x: int(x)),
        '--ext': Const(lambda x: len(x) > 0),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    dfs = parse_input(args['-I'], args['-f'])
    out = args['--output']
    ext = args['--ext']
    model = args['--model']
    weights_dir = args['--weights']
    unique_snps, unique_BADs, merged_df = merge_dfs([x[1] for x in dfs])
    if not args['aggregate']:
        if not os.path.exists(out):
            try:
                os.mkdir(out)
            except Exception:
                print(__doc__)
                exit('Can not create output directory')
                raise
        if model == 'BetaNB':
            models_dict = {}
            for BAD in unique_BADs:
                # FIXME
                models_dict[BAD] = ModelMixture(bad=BAD, left=4, model='BetaNB')
            params = bridge_mixalime.read_dist_from_folder(folder=weights_dir)
            print(params['params'].keys())
            fit_params = params, models_dict
        else:
            try:
                fit_params = check_fit_params_for_BADs(weights_dir,
                                                       unique_BADs)
            except Exception:
                print(__doc__)
                exit('Wrong format weights')
                raise

        if not args['--no-fit']:
            result_dfs = start_process(
                merged_df=merged_df,
                out_path=out,
                dfs=dfs,
                unique_snps=unique_snps,
                fit_params=fit_params,
                model=args['--model']
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
        aggregated_df = aggregate_dfs(merged_df, unique_snps)
        if aggregated_df.empty:
            raise AssertionError('No SNPs left after aggregation')
        maxc_tr = args['--coverage-tr']
        mc_filter_array = np.array(aggregated_df['MAX_COVER'] >= maxc_tr, dtype=np.bool)
        if mc_filter_array.sum() != 0:
            bool_ar_ref, p_val_ref, _, _ = multitest.multipletests(
                aggregated_df[mc_filter_array]["LOGITP_REF"].to_numpy(),
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
