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
from negbin_fit.helpers import init_docopt, make_cover_negative_binom_density, read_stats_df, make_out_path, \
    get_counts_dist_from_df, make_geom_dens
from negbin_fit.visualize import draw_cover_fit, get_callback_plot


def get_rp_from_x(x):
    # m0 = x[0]
    r0 = x[0]
    p0 = x[1]
    w0 = x[2]
    th0 = x[3]
    # r0 = (1 / p0 - 1) * m0
    return r0, p0, w0, th0


# FIXME
def make_log_likelihood_cover(counts_array, cover_left_most, right_most):
    def target(x):
        r0, p0, w0, th0 = get_rp_from_x(x)
        neg_bin_dens = make_cover_negative_binom_density(r0, p0, right_most, cover_left_most, log=False)
        geom_dens = make_geom_dens(th0, cover_left_most, right_most)
        print(r0, p0, w0, th0, -1 * sum(counts_array[k] * (
            np.log((1 - w0) * neg_bin_dens[k] + w0 * geom_dens[k]) if neg_bin_dens[k] != 0 else 0)
                        for k in range(cover_left_most, right_most) if counts_array[k] != 0))
        return -1 * sum(counts_array[k] * (
            np.log((1 - w0) * neg_bin_dens[k] + w0 * geom_dens[k]) if neg_bin_dens[k] != 0 else 0)
                        for k in range(cover_left_most, right_most) if counts_array[k] != 0)

    return target


def calculate_cover_dist_gof():
    return 0


def fit_cover_dist(stats_df, cover_left_most, max_read_count):
    counts_array = get_counts_dist_from_df(stats_df)
    try:
        x = optimize.minimize(fun=make_log_likelihood_cover(counts_array, cover_left_most, max_read_count),
                              x0=np.array([1.5, 0.5, 0.5, 0.8]),
                              bounds=[(0.00000001, 10), (0.01, 0.99), (0, 1), (0.01, 0.99)],)
                              #callback=get_callback_plot(cover_left_most, max_read_count, stats_df))
    except ValueError:
        return 'NaN', 0, 0, 0, 0
    print(x)
    r0, p0, w0, th0 = get_rp_from_x(x.x)
    return r0, p0, w0, th0, calculate_cover_dist_gof()  # TODO: call and save


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
    cover_allele_tr = args['--allele-reads-tr']
    max_read_count = 100
    if not args['visualize']:
        r, p, w, th, gof = fit_cover_dist(df, cover_allele_tr, max_read_count=max_read_count)
        d = {'r0': r, 'p0': p, 'w0': w, 'th0': th, 'gof': gof}
        with open(get_cover_file_path(make_out_path(args['--output'], filename)), 'w') as out:
            json.dump(d, out)
    else:
        d = args['--weights']
    if args['--visualize'] or args['visualize']:
        draw_cover_fit(
            stats_df=df,
            weights_dict=d,
            cover_allele_tr=cover_allele_tr,
            max_read_count=max_read_count
        )
