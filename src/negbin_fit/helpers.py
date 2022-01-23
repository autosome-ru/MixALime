import os
import numpy as np
import json
import re
from docopt import docopt
from schema import SchemaError
from scipy import stats as st
import pandas as pd

alleles = {'ref': 'alt', 'alt': 'ref'}
available_models = ('NB_G', 'BetaNB', 'NB_AS', 'NB_AS_Total')
aggregation_methods = ('logit', 'fisher')


class ParamsHandler:
    def __init__(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self._params = params

    def to_dict(self):
        return self._params

    def __repr__(self):
        return str(self.to_dict())

    def __len__(self):
        return len(self.to_dict())


def make_np_array_path(out, allele, line_fit=False):
    return os.path.join(out, allele + '.' + ('npy' if not line_fit else 'json'))


def parse_input(dfs, filenames):
    if filenames is not None:
        return read_dfs(filenames)
    elif dfs:
        return dfs
    else:
        exit('No input files provided, consider using -I or -f options')
        raise AssertionError


def parse_files_list(filename):
    with open(filename) as inp:
        lines = inp.read().splitlines()
    return [x for x in lines if not x.startswith('#')]


def add_BAD_to_path(out_path, BAD, create=True):
    return make_out_path(out_path, 'BAD{:.2f}'.format(BAD), create=create)


def get_nb_weight_path(out, allele):
    return os.path.join(out, 'NBweights_{}.tsv'.format(allele))


def check_weights_path(weights_path, line_fit):
    return weights_path, {allele: read_weights(line_fit=line_fit,
                                               np_weights_path=weights_path,
                                               allele=alleles[allele]) for allele in alleles}


def read_weights(allele, np_weights_path=None, np_weights_dict=None, line_fit=False):
    if np_weights_path:
        path = make_np_array_path(np_weights_path,
                                  allele, line_fit=line_fit)
        if not line_fit:
            np_weights = np.load(path)
        else:
            with open(path) as r:
                np_weights = json.load(r)
    elif np_weights_dict:
        np_weights = np_weights_dict[allele]
    else:
        raise AssertionError('No numpy fits provided')
    return np_weights


def get_p(BAD):
    return 1 / (BAD + 1)


def get_inferred_mode_w(m, r0, p0, p1, p):
    try:
        return 1 / (1 +
                    np.exp(m * (np.log1p(-p) - np.log(p)) + (m + r0) * (np.log1p(-(1-p*(1-p1))*p0) - np.log1p(-(1 - (1-p)*(1-p1))*p0)))
                    )
    except OverflowError:
        print(m, r0, p0, p1, p)
        raise


def get_p12_from_mu_pc(mu, pc):
    return (
        1 - (1 - pc) * (1 - mu),
        1 - pc * (1 - mu)
    )


def make_inferred_negative_binom_density(m, r0, p0, mu, pc, p, max_c, min_c, fixed_allele, w=None):
    p1, p2 = get_p12_from_mu_pc(mu, pc)
    if fixed_allele == 'ref':
        p1, p2 = p2, p1
    if w is None:
        w = get_inferred_mode_w(m, r0, p0, p1, p)

    p_left = p * p0 * (1 - p2) / (1 - p0 * ((1 - p) * p1 + p * p2))
    p_right = (1 - p) * p0 * (1 - p2) / (1 - p0 * (p * p1 + (1 - p) * p2))
    return make_negative_binom_density(m + r0,
                                       p_left,
                                       w,
                                       max_c,
                                       min_c,
                                       p_right_mode=p_right)


def make_negative_binom_density(r, p_left_mode, w, size_of_counts, left_most, p_right_mode=None):
    if p_right_mode is None:
        p_right_mode = 1 - p_left_mode
    negative_binom_density_array = np.zeros(size_of_counts + 1, dtype=np.float64)
    dist1 = st.nbinom(r, 1 - p_right_mode)  # 1 - p: right mode
    f1 = dist1.pmf
    cdf1 = dist1.cdf
    dist2 = st.nbinom(r, 1 - p_left_mode)  # p: left mode
    f2 = dist2.pmf
    cdf2 = dist2.cdf
    negative_binom_norm = (cdf1(size_of_counts) -
                           (cdf1(left_most - 1) if left_most >= 1 else 0)
                           ) * w + \
                          (cdf2(size_of_counts) -
                           (cdf2(left_most - 1) if left_most >= 1 else 0)
                           ) * (1 - w)
    for k in range(left_most, size_of_counts + 1):
        negative_binom_density_array[k] = \
            (w * f1(k) + (1 - w) * f2(k)) / negative_binom_norm if negative_binom_norm != 0 else 0
    return negative_binom_density_array


def make_prior_fixed_allele_density(params, p, N, allele_tr, fixed_allele, log=True):
    p1, p2 = get_p12_from_mu_pc(params.mu, params.pc)
    if fixed_allele == 'ref':
        p1, p2 = p2, p1

    dist = {
        'nb1': st.nbinom(params.r0, 1 - (1-p)*params.p0 * (1-p1)/(1 - params.p0 * (1 - (1-p1)*(1-p)))),
        'nb2': st.nbinom(params.r0, 1 - p*params.p0 * (1-p1)/(1 - params.p0 * (1 - (1-p1)*p))),
    }

    pmfs = {
        label: dist[label].pmf for label in dist
    }

    cdfs = {
        label: dist[label].cdf for label in dist
    }

    pmf = lambda x: (pmfs['nb1'](x) + pmfs['nb2'](x)) * 0.5

    cdf = lambda x: (cdfs['nb1'](x) + cdfs['nb2'](x)) * 0.5

    norm = cdf(N) - cdf(max(allele_tr - 1, 0))

    density_array = np.zeros(N + 1, dtype=np.float64)
    if not log:
        if norm != 0:
            for k in range(allele_tr, N + 1):
                density_array[k] = \
                    pmf(k) / norm
    else:
        if norm != 0:
            norm = np.log(norm)
            for k in range(N + 1):
                if k < allele_tr:
                    density_array[k] = -np.inf
                else:
                    density_array[k] = np.log(pmf(k)) - norm
        else:
            for k in range(N + 1):
                density_array[k] = -np.inf
    return density_array


def make_binom_density(n, p):
    binom = np.zeros(n + 1)
    if p != 0.5:
        f1 = st.binom(n, p).pmf
        f2 = st.binom(n, 1 - p).pmf
        binom_norm = 1 - sum(0.5 * (f1(k) + f2(k)) for k in [0, 1, 2, 3, 4, n - 4, n - 3, n - 2, n - 1, n])
        for k in range(5, n - 4):
            binom[k] = 0.5 * (f1(k) + f2(k)) / binom_norm
    else:
        f = st.binom(n, p).pmf
        binom_norm = 1 - sum(f(k) for k in [0, 1, 2, 3, 4, n - 4, n - 3, n - 2, n - 1, n])
        for k in range(5, n - 4):
            binom[k] = f(k) / binom_norm
    return binom


def make_out_path(out, name, create=True):
    directory = os.path.join(out, name)
    if not os.path.exists(directory):
        if create:
            os.mkdir(directory)
        else:
            print('No weights found in {}'.format(directory))
            exit(1)
    return directory


def get_counts_dist_from_df(stats_df):
    stats_df['cover'] = stats_df['ref'] + stats_df['alt']
    return [stats_df[stats_df['cover'] == cover]['counts'].sum() for cover in range(stats_df['cover'].max())]


def init_docopt(doc, schema):
    args = docopt(doc)
    try:
        args = schema.validate(args)
    except SchemaError as e:
        print(args)
        print(doc)
        exit('Error: {}'.format(e))
    return args


def read_stats_df(filename):
    try:
        stats = pd.read_table(filename)
        assert set(stats.columns) == {'ref', 'alt', 'counts'}
        for allele in alleles:
            stats[allele] = stats[allele].astype(int)
        return stats, os.path.splitext(os.path.basename(filename))[0]
    except Exception:
        raise AssertionError


def make_cover_negative_binom_density(r, p, size_of_counts, left_most, log=False, draw_rest=False):
    negative_binom_density_array = np.zeros(size_of_counts + 1, dtype=np.float64)
    dist = st.nbinom(r, 1 - p)
    if log:
        f = dist.logpmf
    else:
        f = dist.pmf
    cdf = dist.cdf
    negative_binom_norm = cdf(size_of_counts) - (cdf(left_most - 1) if left_most >= 1 else 0)
    for k in range(0, size_of_counts + 1):
        negative_binom_density_array[k] = \
            f(k) if k >= left_most or draw_rest else (-np.inf if log else 0)
    return negative_binom_density_array - np.log(negative_binom_norm) if log else negative_binom_density_array / negative_binom_norm


def make_geom_dens(p, a, b, draw_rest=False):
    geom_density_array = np.zeros(b + 1, dtype=np.float64)
    dist = st.geom(1-p)
    f = dist.pmf
    cdf = dist.cdf
    geom_norm = cdf(b) - (cdf(a - 1) if a >= 1 else 0)
    for k in range(0, b + 1):
        geom_density_array[k] = \
            f(k) if k >= a or draw_rest else 0
    return geom_density_array / geom_norm


def combine_densities(negbin_dens, geom_dens, w, frac, p, allele_tr=5, only_negbin=False):
    comb_dens = w * geom_dens + (1 - w) * negbin_dens
    # for k in range(allele_tr * 2, len(comb_dens)):
    #     comb_dens[k] *= (1 + frac * (get_norm(p, k, allele_tr) + get_norm(1 - p, k, allele_tr)))
    if only_negbin:
        return (1 - w) * negbin_dens / comb_dens.sum()
    else:
        return comb_dens / comb_dens.sum()


def make_line_negative_binom_density(fix_c, params, p, N, allele_tr, fixed_allele, log=True):
    neg_bin_dens = make_inferred_negative_binom_density(fix_c, params.r0, params.p0, params.mu, params.pc, p, N, allele_tr, fixed_allele)
    with np.errstate(divide='ignore'):
        return np.log(neg_bin_dens) if log else neg_bin_dens


def stats_df_to_numpy(stats_df, min_tr, max_tr):
    rv = np.zeros([max_tr + 1, max_tr + 1], dtype=np.int_)
    for k in range(min_tr, max_tr + 1):
        for m in range(min_tr, max_tr + 1):
            slice = stats_df[(stats_df['ref'] == k) & (stats_df['alt'] == m)]
            if not slice.empty:
                rv[k, m] = slice['counts']
    return rv


def rmsea_gof(stat, df, norm):
    if norm <= 1:
        return 0
    else:
        # if max(stat - df, 0) / (df * (norm - 1)) < 0:
        #     print(stat, df)
        score = np.sqrt(max(stat - df, 0) / (df * (norm - 1)))
    return score


def calculate_gof_for_point_fit(counts_array, expected, norm, number_of_params, left_most):
    observed = counts_array.copy()
    observed[:left_most] = 0

    idxs = (observed != 0) & (expected != 0)
    if idxs.sum() <= number_of_params + 1:
        return 0
    df = idxs.sum() - 1 - number_of_params
    stat = np.sum(observed[idxs] * (np.log(observed[idxs]) - np.log(expected[idxs]))) * 2
    return rmsea_gof(stat, df, norm)


def calculate_overall_gof(stats_df, density_func, params, main_allele, min_tr, max_tr, num_params=None):
    if num_params is None:
        num_params = len(params)
    observed = stats_df_to_numpy(stats_df, min_tr, max_tr)
    assert main_allele in ('ref', 'alt')
    if main_allele == 'alt':
        observed = observed.transpose()
    expected = np.zeros([max_tr + 1, max_tr + 1], dtype=np.int_)
    point_gofs = {}
    for fix_c in range(min_tr, max_tr + 1):
        observed_for_fix_c = observed[:, fix_c]
        norm = observed_for_fix_c.sum()
        expected[:, fix_c] = density_func(fix_c) * norm
        point_gofs[str(fix_c)] = calculate_gof_for_point_fit(observed_for_fix_c, expected[:, fix_c], norm, num_params, min_tr)
    overall_gof = calculate_gof_for_point_fit(observed.flatten(), expected.flatten(), observed.sum(), num_params, min_tr)
    return point_gofs, overall_gof


def merge_dfs(dfs):
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df['key'].unique(), merged_df['BAD'].unique(), merged_df


def get_counts_column(allele, for_what='counts'):
    if for_what == 'counts':
        result = allele + '_counts'
    elif for_what == 'pval':
        result = 'PVAL_' + allele
    elif for_what == 'es':
        result = 'ES_' + allele
    else:
        raise ValueError
    return result.upper()


def get_pvalue_file_path(out_path, df_name):
    return os.path.join(out_path, df_name + '.pvalue_table')


def get_required_df_fields():
    return '#CHROM', 'POS', 'ID', 'REF', 'ALT'


def get_key(row):
    return '@'.join(map(str, [row[field] for field in get_required_df_fields()]))


def read_dfs(filenames):
    result = []
    print('Reading tables...')
    for filename in filenames:
        df = read_df(filename)
        if df is not None:
            result.append(df)
    return result


def read_df(filename):
    try:
        df = pd.read_table(filename)
        df = df[df['REF_COUNTS'] >= 5]
        df = df[df['ALT_COUNTS'] >= 5]
        if df.empty:
            print('No SNPs found in {}'.format(filename))
            return None
        df['key'] = df.apply(get_key, axis=1)
        df['fname'] = filename
    except Exception:
        raise ValueError('Cannot read {}'.format(filename))
    return os.path.splitext(os.path.basename(filename))[0], df


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
