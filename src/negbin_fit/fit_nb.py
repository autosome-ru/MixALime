"""
Usage:
    negbin_fit [-O <dir> |--output <dir>] [-n | --no-fit] [-q | --quiet] [--allele-reads-tr <int>] [--visualize] [-l | --line-fit] [--max-read-count <int>] [-c | --collect] [--cover-list <list>] <file> ...
    negbin_fit -h | --help

Arguments:
    <file>            Path to input file in tsv format with columns: alt ref counts.
    <bad>             BAD value (can be decimal)
    <int>             Non negative integer
    <dir>             Directory for fitted weights
    <list>            List of slices to visualize


Options:
    -h, --help                              Show help
    -q, --quiet                             Suppress log messages
    -O <path>, --output <path>              Output directory for obtained fits.
    -c, --collect                           Collect BAD-wise stats, use existing stats if option is not provided
    -n, --no-fit                            Skip p-value calculation (use to visualize results)
    --allele-reads-tr <int>                 Allelic reads threshold. Input SNPs will be filtered by ref_read_count >= x and alt_read_count >= x. [default: 5]
    --visualize                             Perform visualization
    -l, --line-fit                          Fit all the data with line
    --max-read-count <int>                  Max read count for visualization [default: 50]
    --cover-list <list>                     List of covers to visualize [default: 10,20,30,40,50]
"""
import json
import os
import re

import numpy as np
import pandas as pd
from negbin_fit.helpers import alleles, make_np_array_path, get_p, init_docopt, \
    make_negative_binom_density, make_out_path, make_line_negative_binom_density, calculate_gof_for_point_fit, \
    ParamsHandler, calculate_overall_gof, check_weights_path, add_BAD_to_path, merge_dfs, read_dfs, get_counts_column
from negbin_fit.neg_bin_weights_to_df import main as convert_weights
from schema import And, Const, Schema, Use, Or
from scipy import optimize
from tqdm import tqdm


def make_scaled_counts(stats_pandas_dataframe, main_allele, max_cover_in_stats):
    counts_array = np.zeros(max_cover_in_stats + 1, dtype=np.int64)
    nonzero_set = set()

    for index, row in stats_pandas_dataframe.iterrows():
        k, SNP_counts = row[main_allele], row['counts']
        if k > max_cover_in_stats:
            continue
        nonzero_set.add(k)
        counts_array[k] += SNP_counts
    return counts_array, nonzero_set


def fit_negative_binom(n, counts_array, fix_c, BAD, left_most):
    try:
        x = optimize.minimize(fun=make_log_likelihood(n, counts_array, BAD, left_most),
                              x0=np.array([fix_c, 0.5]),
                              bounds=[(0.00001, None), (0, 1)])
    except ValueError:
        return 'NaN', 0
    r, w = x.x
    density = make_negative_binom_density(r, get_p(BAD), w, len(counts_array) - 1, left_most)
    norm = counts_array[left_most:].sum()
    expected = density * norm
    return x, calculate_gof_for_point_fit(counts_array, expected, norm, 1 if BAD == 1 else 2, left_most)


def make_log_likelihood(n, counts_array, BAD, left_most):
    def target(x):
        r = x[0]
        w = x[1]
        neg_bin_dens = make_negative_binom_density(r, get_p(BAD), w, len(counts_array), left_most)
        return -1 * sum(counts_array[k] * (
            np.log(neg_bin_dens[k]) if neg_bin_dens[k] != 0 else 0)
                        for k in range(left_most, n) if counts_array[k] != 0)

    return target


def get_line_params_from_x(x):
    return ParamsHandler(r0=x[0], p0=x[1], w0=x[2], th0=x[3])


def make_likelihood_as_line(stats, main_allele, upper_bound, N, allele_tr=5, BAD=1):
    p = 1 / (BAD + 1)

    def target(x):
        params = get_line_params_from_x(x)
        result = 0
        for fix_c in range(allele_tr, upper_bound + 1):
            stats_filtered, counts_array = preprocess_stats(stats, fix_c, N, main_allele, allele_tr)
            if stats_filtered is None:
                continue
            neg_bin_dens = make_line_negative_binom_density(fix_c, params, p, N, allele_tr)
            result += -1 * sum(counts_array[k] * (
                    (neg_bin_dens[k]
                     if neg_bin_dens[k] != -np.inf else 0) + 0)
                               for k in range(allele_tr, N) if counts_array[k] != 0)  # / \
            # sum(counts_array[k] for k in range(allele_tr, N) if counts_array[k] != 0)
        return result

    return target


def fit_negative_binom_as_line(stats_df, main_allele, upper_bound, N, allele_tr, BAD):
    try:
        x = optimize.minimize(fun=make_likelihood_as_line(stats_df, main_allele,
                                                          upper_bound=upper_bound,
                                                          N=N,
                                                          allele_tr=allele_tr,
                                                          BAD=BAD),
                              x0=np.array([-2, 0.95, 0.75, 0.67]),
                              bounds=[(-4, 10), (0.01, 0.99), (0, 1), (0.01, 0.99)])
    except ValueError:
        return 'NaN', 'NaN', 0
    params = get_line_params_from_x(x.x)
    density_func = lambda fix_c: make_line_negative_binom_density(fix_c, params, get_p(BAD), N,
                                                                  allele_tr, log=False)
    point_gofs, overall_gof = calculate_overall_gof(stats_df, density_func, params, main_allele, allele_tr, N)
    return params, point_gofs, overall_gof


def preprocess_stats(stats, fix_c, N, main_allele, allele_tr):
    stats_filtered = stats[stats[alleles[main_allele]] == fix_c]
    try:
        counts, set_of_nonzero_n = make_scaled_counts(stats_filtered, main_allele, N)
    except ValueError:
        counts, set_of_nonzero_n = [], set()

    if len(set_of_nonzero_n) == 0 or counts.sum() < max(set_of_nonzero_n) - allele_tr:
        return None, None
    return stats_filtered, counts


def fit_neg_bin_for_allele(stats, main_allele, BAD=1, allele_tr=5, upper_bound=100, line_fit=False, max_read_count=100):
    print('Fitting {} distribution BAD={}...'.format(main_allele.upper(), BAD))
    N = min(max(stats[main_allele]), max_read_count)
    if not line_fit:
        save_array = np.zeros((upper_bound + 1, 4), dtype=np.float_)
        for fix_c in tqdm(range(allele_tr, upper_bound + 1)):
            stats_filtered, counts = preprocess_stats(stats,
                                                      fix_c, N,
                                                      main_allele, allele_tr)
            if stats_filtered is None:
                continue
            weights, gof = fit_negative_binom(N, counts, fix_c, BAD, allele_tr)
            save_array[fix_c, :2] = weights.x
            save_array[fix_c, 2] = weights.success
            save_array[fix_c, 3] = gof
        return save_array
    else:
        params, point_gofs, gof = fit_negative_binom_as_line(stats,
                                                             main_allele,
                                                             upper_bound=upper_bound,
                                                             N=N,
                                                             allele_tr=allele_tr,
                                                             BAD=BAD,
                                                             )
        return {**params.to_dict(), 'point_gofs': point_gofs, 'gof': gof}


def calculate_cover_dist_gof():
    return 0


def main(stats, out=None, BAD=1, allele_tr=5, line_fit=False, max_read_count=100):
    d = {}
    for main_allele in alleles:
        save_array = fit_neg_bin_for_allele(stats,
                                            main_allele,
                                            BAD=BAD,
                                            line_fit=line_fit,
                                            upper_bound=200,
                                            allele_tr=allele_tr,
                                            max_read_count=max_read_count)

        d[main_allele] = save_array
        if not line_fit:
            np.save(make_np_array_path(out, main_allele), save_array)
        else:
            with open(make_np_array_path(out, main_allele, line_fit=line_fit), 'w') as f:
                json.dump(save_array, f)
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


def parse_cover_list(list_as_string):
    cover_list = [int(x) for x in list_as_string.split(',') if int(x) > 0]
    assert len(cover_list) > 0
    return cover_list


def open_stats_df(out_path):
    return pd.read_table(get_stats_df_path(out_path), header=None, names=['ref', 'alt', 'counts'])


def get_stats_df_path(out_path):
    return os.path.join(out_path, 'stats.tsv')


def collect_stats_df(df, out_path, BAD):
    # sum_df = [[get_counts_column(x) for x in alleles]]
    out_t = df[df['BAD'] == BAD].groupby([get_counts_column(x) for x in alleles]).size().reset_index(name='counts')
    out_t.fillna(0, inplace=True)
    out_t.columns = ['ref', 'alt', 'counts']
    out_t.to_csv(get_stats_df_path(out_path),
                 index=False,
                 header=None,
                 sep='\t')
    return out_t


def check_output(x):
    if not os.path.exists(x):
        os.mkdir(x)
    return True


def start_fit():
    schema = Schema({
        '<file>': And(
            Const(lambda x: sum(os.path.exists(y) for y in x),
                  error='Input file(s) should exist'),
            Use(read_dfs, error='Wrong format stats file')
        ),
        '--output': Or(
            Const(lambda x: x is None),
            And(
                Const(check_output, error="No output path exist"),
                Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
            )
        ),
        '--allele-reads-tr': And(
            Use(int),
            Const(lambda x: x >= 0), error='Allelic reads threshold must be a non negative integer'
        ),
        '--cover-list': Use(parse_cover_list, error='Wrong format cover list'),
        '--max-read-count': And(
            Use(int),
            Const(lambda x: x > 0),
            error='Max read count threshold must be a positive integer'
        ),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    dfs = args['<file>']
    allele_tr = args['--allele-reads-tr']
    out_path = args['--output']
    line_fit = args['--line-fit']
    # if line_fit and BAD != 1:
    #     print('Line fit for BAD != 1 not implemented')
    #     exit(1)
    _, unique_BADs, merged_df = merge_dfs([x[1] for x in dfs])
    max_read_count = 100
    print('{} unique BADs detected'.format(len(unique_BADs)))
    for BAD in sorted(unique_BADs):
        if out_path is None:
            out_path = make_out_path('./', dfs[0][0])
        out = add_BAD_to_path(out_path, BAD)
        if args['--collect']:
            print('Collecting stats file...')
            stats_df = collect_stats_df(merged_df, out, BAD)
        else:
            print('Using existing stats...')
            stats_df = open_stats_df(out)

        if not args['--no-fit']:

            d = main(stats_df,
                     out=out,
                     BAD=BAD,
                     line_fit=line_fit,
                     allele_tr=allele_tr,
                     max_read_count=max_read_count)
            if not line_fit:
                convert_weights(in_df=stats_df,
                                np_weights_dict=d,
                                out_path=out)
        else:
            try:
                _, d = check_weights_path(out, line_fit=line_fit)

                stats_df = open_stats_df(out)
            except Exception:
                print(__doc__)
                exit('Wrong format weights')
                raise
        if args['--visualize']:
            from negbin_fit.visualize import main as visualize
            visualize(
                stats=stats_df,
                weights_dict=d,
                line_fit=line_fit,
                cover_list=args['--cover-list'],
                max_read_count=args['--max-read-count'],
                out=out,
                BAD=BAD,
                allele_tr=allele_tr)
