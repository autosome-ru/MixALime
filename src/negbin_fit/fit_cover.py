"""
Usage:
    cover_fit <file> (-b <bad> | --bad <bad>) [-O <dir> |--output <dir>] [-q | --quiet] [--allele-reads-tr <int>] [--visualize]
    cover_fit -h | --help
    cover_fit visualize <file> (-b <bad> | --bad <bad>) (-w <dir> |--weights <dir>)  [--allele-reads-tr <int>]

Arguments:
    <file>            Path to input file in tsv format with columns: alt ref counts.
    <int>             Non negative integer
    <dir>             Directory for fitted weights
    <bad>             BAD value (can be decimal)

Options:
    -h, --help                              Show help.
    -q, --quiet                             Suppress log messages.
    -O <path>, --output <path>              Output directory for obtained fits. [default: ./]
    -w <path>, --weights <path>             Directory with obtained fits
    --allele-reads-tr <int>                 Allelic reads threshold. Input SNPs will be filtered by ref_read_count >= x and alt_read_count >= x. [default: 5]
    --visualize                             Perform visualization
    -b <bad>, --bad <bad>                   BAD value used in fit (can be decimal)
"""
import json
import os

from negbin_fit.fit_nb import convert_string_to_float
from schema import Schema, And, Const, Use, Or
from scipy import optimize
import numpy as np
from negbin_fit.helpers import init_docopt, make_cover_negative_binom_density, read_stats_df, make_out_path, \
    get_counts_dist_from_df, make_geom_dens, combine_densities
from negbin_fit.visualize import draw_cover_fit, get_callback_plot


def get_rp_from_x(x):
    # m0 = x[0]
    r0 = x[0]
    p0 = x[1]
    w0 = x[2]
    th0 = x[3]
    frac = x[4]
    # r0 = (1 / p0 - 1) * m0
    return r0, p0, w0, th0, frac


# FIXME
def make_log_likelihood_cover(counts_array, cover_left_most, right_most, BAD=1):
    def target(x):
        r0, p0, w0, th0, frac = get_rp_from_x(x)
        neg_bin_dens = make_cover_negative_binom_density(r0, p0, right_most, cover_left_most, log=False)
        geom_dens = make_geom_dens(th0, cover_left_most, right_most)
        comb_dens = combine_densities(neg_bin_dens, geom_dens, w0, 0.9, 1 / (BAD + 1), allele_tr=5)
        comb_dens = np.log(comb_dens)
        ret_v = -1 * sum(counts_array[k] * (
            comb_dens[k] if neg_bin_dens[k] != -np.inf else 0)
                        for k in range(cover_left_most, right_most) if counts_array[k] != 0)
        print(r0, p0, w0, th0, ret_v)
        return ret_v

    return target


def calculate_cover_dist_gof():
    return 0


def fit_cover_dist(stats_df, cover_left_most, max_read_count, BAD=1):
    counts_array = get_counts_dist_from_df(stats_df)
    try:
        x = optimize.minimize(fun=make_log_likelihood_cover(counts_array, cover_left_most, max_read_count, BAD),
                              x0=np.array([1.5, 0.5, 0.5, 0.5, 0.9]),
                              bounds=[(0.00000001, 10), (0.01, 0.999), (0, 1), (0.1, 0.9), (0, 1)],)
                              #callback=get_callback_plot(cover_left_most, max_read_count, stats_df, BAD=BAD))
    except ValueError:
        return 'NaN', 0, 0, 0, 0
    print(x)
    r0, p0, w0, th0, frac = get_rp_from_x(x.x)
    return r0, p0, w0, th0, frac, calculate_cover_dist_gof()  # TODO: call and save


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
        '--bad': And(
            Use(convert_string_to_float, error='Wrong format BAD'),
            Const(lambda x: x >= 1, error='BAD must be >= 1')
        ),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    df, filename = args['<file>']
    cover_allele_tr = args['--allele-reads-tr']
    max_read_count = 100
    if not args['visualize']:
        r, p, w, th, frac, gof = fit_cover_dist(df, cover_allele_tr, max_read_count=max_read_count, BAD=args['--bad'])
        d = {'r0': r, 'p0': p, 'w0': w, 'th0': th, 'frac': frac, 'gof': gof}
        with open(get_cover_file_path(make_out_path(args['--output'], filename)), 'w') as out:
            json.dump(d, out)
    else:
        d = args['--weights']
    if args['--visualize'] or args['visualize']:
        draw_cover_fit(
            stats_df=df,
            weights_dict=d,
            cover_allele_tr=cover_allele_tr,
            max_read_count=max_read_count,
            BAD=args['--bad']
        )
