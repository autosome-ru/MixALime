"""
Usage:
    cover_fit <file> [-O <dir> |--output <dir>] [-q | --quiet] [--allele-reads-tr <int>] [--visualize]
    cover_fit -h | --help
    cover_fit visualize <file> (-w <dir> |--weights <dir>)  [--allele-reads-tr <int>]

Arguments:
    <file>            Path to input file in tsv format with columns: alt ref counts.
    <int>             Non negative integer
    <dir>             Directory for fitted weights

Options:
    -h, --help                              Show help.
    -q, --quiet                             Suppress log messages.
    -O <path>, --output <path>              Output directory for obtained fits. [default: ./]
    -w <path>, --weights <path>             Directory with obtained fits
    --allele-reads-tr <int>                 Allelic reads threshold. Input SNPs will be filtered by ref_read_count >= x and alt_read_count >= x. [default: 5]
    --visualize                             Perform visualization
"""
import json
import os
from schema import Schema, And, Const, Use, Or
from scipy import optimize
import numpy as np
from negbin_fit.helpers import init_docopt, make_negative_binom_density, read_stats_df, make_out_path, \
    get_counts_dist_from_df
from negbin_fit.visualize import draw_cover_fit


# FIXME
def make_log_likelihood_cover(counts_array, left_most):
    def target(x):
        r0 = x[0]
        p0 = x[1]
        neg_bin_dens = make_negative_binom_density(r0, p0, 0, len(counts_array), left_most)
        print(neg_bin_dens, x)
        return -1 * sum(counts_array[k] * (
            np.log(neg_bin_dens[k]) if neg_bin_dens[k] != 0 else 0)
                        for k in range(left_most, len(counts_array)) if counts_array[k] != 0)

    return target


def calculate_cover_dist_gof():
    return 0


def fit_cover_dist(stats_df, left_most):
    counts_array = get_counts_dist_from_df(stats_df)
    try:
        x = optimize.minimize(fun=make_log_likelihood_cover(counts_array, left_most * 2),
                              x0=np.array([10, 0.5]),
                              bounds=[(0.5, None), (0.01, 0.99)])
    except ValueError:
        return 'NaN', 0
    print(x)
    r0, p0 = x.x
    return r0, p0, calculate_cover_dist_gof()  # TODO: call and save


def get_cover_file_path(dir_path):
    return os.path.join(dir_path, 'cover.json')


def read_cover_weights(weights_path):
    with open(get_cover_file_path(weights_path), 'r') as f:
        return json.load(f)


def main():
    schema = Schema({
        '<file>': And(
            Const(os.path.exists, error='Input file should exist'),
            Use(read_stats_df, error='Wrong format stats file')
        ),
        '--output': And(
            Const(os.path.exists),
            Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
        ),
        '--allele-reads-tr': And(
            Use(int),
            Const(lambda x: x >= 0), error='Allelic reads threshold must be a non negative integer'
        ),
        '--weights': Or(
            Const(lambda x: x is None),
            And(
                Const(os.path.exists),
                Const(lambda x: os.access(x, os.W_OK), error='No write permissions'),
                Use(read_cover_weights, error='Invalid weights file')
            )),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    df, filename = args['<file>']
    allele_tr = args['--allele-reads-tr']
    if not args['visualize']:
        r, p, gof = fit_cover_dist(df, allele_tr)
        d = {'r0': r, 'p0': p, 'gof': gof}
        with open(get_cover_file_path(make_out_path(args['--output'], filename)), 'w') as out:
            json.dump(d, out)
    else:
        d = args['--weights']
    if args['--visualize'] or args['visualize']:
        draw_cover_fit(
            stats_df=df,
            weights_dict=d,
            allele_tr=allele_tr
        )
