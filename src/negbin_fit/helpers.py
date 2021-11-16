import os
import numpy as np
import json
from docopt import docopt
from schema import SchemaError
from scipy import stats as st
import pandas as pd

alleles = {'ref': 'alt', 'alt': 'ref'}


def make_np_array_path(out, allele, line_fit=False):
    return os.path.join(out, allele + '.' + ('npy' if not line_fit else 'json'))


def get_nb_weight_path(out, allele):
    return os.path.join(out, 'NBweights_{}.tsv'.format(allele))


def read_weights(allele, np_weights_path=None, np_weights_dict=None, line_fit=False):
    if np_weights_path:
        path = make_np_array_path(np_weights_path, allele, line_fit=line_fit)
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


def make_inferred_negative_binom_density(m, r0, p0, p, max_c, min_c):
    return make_negative_binom_density(m + r0,
                                       p * p0,
                                       1 / (1 +
                                            (p * p0) ** m * (1 - p0) ** r0 / (1 - p0 * (1 - p)) ** (m + r0)
                                            ),
                                       max_c,
                                       min_c,
                                       p2=(1 - p) * p0)


def make_negative_binom_density(r, p, w, size_of_counts, left_most, p2=None):
    if p2 is None:
        p2 = p
    negative_binom_density_array = np.zeros(size_of_counts + 1, dtype=np.float64)
    dist1 = st.nbinom(r, 1 - (1 - p2))  # 1 - p right mode
    f1 = dist1.pmf
    cdf1 = dist1.cdf
    dist2 = st.nbinom(r, 1 - p)  # p left mode
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


def make_out_path(out, name):
    directory = os.path.join(out, name)
    if not os.path.exists(directory):
        os.mkdir(directory)
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


def get_norm(p, N, trim_cover):
    result = 0
    current_multiplier = 1
    denominator_multiplier = 1
    for k in range(trim_cover):
        result += current_multiplier * np.power(p, N - k) * np.power(1 - p, k) / denominator_multiplier
        current_multiplier *= int(N - k)
        denominator_multiplier *= k + 1

    return -result


def combine_densities(negbin_dens, geom_dens, w, frac, p, allele_tr=5, only_negbin=False):
    comb_dens = w * geom_dens + (1 - w) * negbin_dens
    # for k in range(allele_tr * 2, len(comb_dens)):
    #     comb_dens[k] *= (1 + frac * (get_norm(p, k, allele_tr) + get_norm(1 - p, k, allele_tr)))
    if only_negbin:
        return (1 - w) * negbin_dens / comb_dens.sum()
    else:
        return comb_dens / comb_dens.sum()
