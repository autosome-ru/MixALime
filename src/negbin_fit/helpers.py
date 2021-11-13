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


def make_negative_binom_density(r, p, w, size_of_counts, left_most):
    negative_binom_density_array = np.zeros(size_of_counts + 1, dtype=np.float64)
    dist1 = st.nbinom(r, p)
    f1 = dist1.pmf
    cdf1 = dist1.cdf
    dist2 = st.nbinom(r, 1 - p)
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
