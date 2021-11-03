"""
Usage:
    negbin_fit <file> (-b <bad> | --bad <bad>) [-O <path> |--output <path>] [-q | --quiet] [--allele-reads-tr <int>] [--visualize] [-r | --readable]
    negbin_fit -h | --help

Arguments:
    <file>            Path to input file in tsv format with columns:
                      chr pos ID ref_base alt_base ref_read_count alt_read_count.
    <bad>             BAD value (can be decimal)
    <int>             Non negative integer


Options:
    -h, --help                              Show help.
    -q, --quiet                             Suppress log messages.
    -O <path>, --output <path>              Output directory for obtained fits. [default: ./]
    -b <bad>, --bad <bad>                   BAD value used in fit (can be decimal)
    --allele-reads-tr <int>                 Allelic reads threshold. Input SNPs will be filtered by ref_read_count >= x and
                                            alt_read_count >= x. [default: 5]
    --visualize                             Perform visualization
    -r, --readable                          Save tsv files of fitted distributions
"""
import os
import re
import numpy as np
import pandas as pd
from docopt import docopt
from tqdm import tqdm
from schema import SchemaError, And, Const, Schema, Use
from scipy import optimize, stats as st
from negbin_fit.helpers import alleles, make_np_array_path, get_p


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
    # plot_norm = (cdf1(size_of_counts) -
    #              cdf1(4)
    #              ) * w + \
    #             (cdf2(size_of_counts) -
    #              cdf2(4)
    #              ) * (1 - w)
    for k in range(left_most, size_of_counts + 1):
        negative_binom_density_array[k] = \
            (w * f1(k) + (1 - w) * f2(k)) / negative_binom_norm if negative_binom_norm != 0 else 0
        # if for_plot:
        #     negative_binom_density_array[k] = (w * f1(k) + (1 - w) * f2(k)) / plot_norm
        # else:
        #     negative_binom_density_array[k] = (w * f1(k) + (1 - w) * f2(k)) / negative_binom_norm
    return negative_binom_density_array


def make_scaled_counts(stats_pandas_dataframe, main_allele, max_cover_in_stats):
    counts_array = np.zeros(max_cover_in_stats + 1, dtype=np.int64)
    nonzero_set = set()

    for index, row in stats_pandas_dataframe.iterrows():
        k, SNP_counts = row[main_allele], row['counts']
        nonzero_set.add(k)
        counts_array[k] += SNP_counts
    return counts_array, nonzero_set


def fit_negative_binom(n, counts_array, fix_c, BAD, q_left, left_most):
    try:
        x = optimize.minimize(fun=make_log_likelihood(n, counts_array, BAD, left_most),
                              x0=np.array([fix_c, 0.5]),
                              bounds=[(0.00001, None), (0, 1)])
    except ValueError:
        return 'NaN', 0
    r, w = x.x
    return x, calculate_gof(counts_array, w, r, BAD, q_left, left_most)


def make_log_likelihood(n, counts_array, BAD, left_most):
    def target(x):
        r = x[0]
        w = x[1]
        neg_bin_dens = make_negative_binom_density(r, get_p(BAD), w, len(counts_array), left_most)
        return -1 * sum(counts_array[k] * (
            np.log(neg_bin_dens[k]) if neg_bin_dens[k] != 0 else 0)
                        for k in range(left_most, n) if counts_array[k] != 0)

    return target


def calculate_gof(counts_array, w, r, BAD, q_left, left_most):
    observed = counts_array.copy()
    observed[:q_left] = 0
    norm = observed.sum()
    expected = make_negative_binom_density(r, get_p(BAD), w, len(observed) - 1, left_most) * norm

    idxs = (observed != 0) & (expected != 0)
    if idxs.sum() < 3:
        return 0
    df = idxs.sum() - (3 if BAD != 1 else 2)

    stat = np.sum(observed[idxs] * np.log(observed[idxs] / expected[idxs])) * 2

    if norm <= 1:
        return 0
    else:
        if max(stat - df, 0) / (df * (norm - 1)) < 0:
            print(stat, df)
        score = np.sqrt(max(stat - df, 0) / (df * (norm - 1)))

    return score


def fit_neg_bin_for_allele(stats, main_allele, BAD=1, allele_tr=5, upper_bound=200):
    print('Fitting {} distribution...'.format(main_allele.upper()))
    fixed_allele = alleles[main_allele]
    save_array = np.zeros((upper_bound + 1, 4), dtype=np.float_)
    for fix_c in tqdm(range(allele_tr, upper_bound)):
        stats_filtered = stats[stats[fixed_allele] == fix_c]
        try:
            max_cover = max(stats[main_allele].to_list())
            counts, set_of_nonzero_n = make_scaled_counts(stats_filtered, main_allele, max_cover)
        except ValueError:
            counts, set_of_nonzero_n = [], set()

        if len(set_of_nonzero_n) == 0 or counts.sum() < max(set_of_nonzero_n) - 5:
            continue

        left_most = 5
        q_left = 5
        right_most = len(counts) - 1

        weights, gof = fit_negative_binom(right_most, counts, fix_c, BAD, q_left, left_most)
        save_array[fix_c, :2] = weights.x
        save_array[fix_c, 2] = weights.success
        save_array[fix_c, 3] = gof
    return save_array


def read_stats_df(filename):
    try:
        stats = pd.read_table(filename)
        assert set(stats.columns) == {'ref', 'alt', 'counts'}
        for allele in alleles:
            stats[allele] = stats[allele].astype(int)
        return stats, os.path.splitext(os.path.basename(filename))[0]
    except Exception:
        raise AssertionError


def make_out_path(out, name):
    directory = os.path.join(out, name)
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory


def main(stats, out=None, BAD=1, allele_tr=5):
    d = {}
    for main_allele in alleles:
        save_array = fit_neg_bin_for_allele(stats,
                                            main_allele,
                                            BAD=BAD,
                                            upper_bound=200,
                                            allele_tr=allele_tr)

        d[main_allele] = save_array
        np.save(make_np_array_path(out, main_allele), save_array)
    return d


def convert_string_to_float(bad_str):
    matcher = re.match(r'^(\d)+/(\d)+$', bad_str)
    if matcher and matcher[0]:
        try:
            return int(matcher[1]) / int(matcher[2])
        except ValueError:
            return False
    else:
        try:

            return float(bad_str)
        except ValueError:
            return False


def start_fit():
    args = docopt(__doc__)
    schema = Schema({
        '<file>': And(
            Const(os.path.exists, error='Input file should exist'),
            Use(read_stats_df, error='Wrong format stats file')
        ),
        '--bad': And(
            Use(convert_string_to_float, error='Wrong format BAD'),
            Const(lambda x: x >= 1, error='BAD must be >= 1')
        ),
        '--output': And(
                Const(os.path.exists),
                Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
        ),
        '--allele-reads-tr': And(
            Use(int),
            Const(lambda x: x >= 0), error='Allelic reads threshold must be a non negative integer'
        ),
        str: bool
    })
    try:
        args = schema.validate(args)
    except SchemaError as e:
        print(__doc__)
        exit('Error: {}'.format(e))
    df, filename = args['<file>']
    allele_tr = args['--allele-reads-tr']
    BAD = args['--bad']
    out_path = make_out_path(args['--output'], filename)
    d = main(df,
             out=out_path,
             BAD=BAD,
             allele_tr=allele_tr)
    if args['--readable'] or args['--visualize']:
        from negbin_fit.neg_bin_weights_to_df import main as convert_weights
        convert_weights(in_df=df,
                        np_weights_dict=d,
                        out_path=out_path)
    if args['--visualize']:
        from negbin_fit.visualize import main as visualize
        visualize(
            stats=df,
            np_weights_dict=d,
            out=out_path, BAD=BAD, allele_tr=allele_tr)
