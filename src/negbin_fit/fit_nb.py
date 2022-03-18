"""
Usage:
    negbin_fit [options] (-O <dir> -m <model>)
    negbin_fit collect (-I <file>... | -f <file-list>) (-O <dir> -m <model>)

Arguments:
    <file>            Path to input file in tsv format with columns: alt ref counts
    <int>             Non negative integer
    <string>          String of states separated with "," (to provide fraction use "/", e.g. 4/3). Each state must be >= 1
    <dir>             Directory with fitted weights
    <list>            List of positive integers "int1,int2" or slice "start:end:step"
    <file-list>       File with filenames of input file on each line
    <model>           String any of (NB_AS, BetaNB, NB_AS_Total, line, window, slices)
    <conc>            String any of (line, intercept)
    <dist>            String any of (NB, BetaNB)
    <bh>              String any of (both, right)
    <ext>             Extension, non-empty string

Required:
    -I <file>...                            Path to input file(s)
    -f <file-list>                          File with filenames of input file on each line
    -O <path>, --output <path>              Output directory for obtained fits.
    -m <model>, --model <model>             Model to fit data with [default: NB_AS_Total]

Optional:
    -h, --help                                  Show help
    -q, --quiet                                 Suppress log messages
    --states <string>                           Set of states to perform fit on
    -l <int>, --reads-left-tr <int>             Left allelic reads threshold. Input SNPs will be filtered by
                                                ref_read_count >= x and alt_read_count >= x. [default: 5]
    -r <int>, --reads-right-tr <int>            Right main allelic reads threshold. Input SNPs will be filtered by
                                                ref_read_count <= x. If 'inf', then no threshould is applied.[default: inf]
    -R <int>, --reads-right-tr-alt <int>        Right alternative allelic reads threshold. Input SNPs will be filtered by
                                                alt_read_count <= x. If 'inf', then no threshold is applied. If 'none', 
                                                then it is the same as reads-right-tr. [default: none]
    -c <conc>, --concentration <conc>           Concentration parameter for ModelLine and ModelWindow model [default: line]
    -d <dist>, --distribution <dist>            Distribution to be used in ModelLine, ModelWindow or ModelMixtures models. Can be either BetaNB or NB [default: BetaNB]
    -w <int>, --window_size <int>               Minimal window size for ModelWindow [default: 1000]
    -s <int>, --min_slices <int>                Minimal number of slices per window for ModelWindow [default: 10]
    -b <bh>, --window_behavior <bh>             If 'both', then window is expanded into both directions. If 'right', then it expands only to the right as long as it is possible [default: both]

Visualize:
    -n, --no-fit                            Skip fitting procedure (use to visualize results)
    --visualize                             Perform visualization
    --max-read-count <int>                  Max read count for visualization [default: 50]
    --cover-list <list>                     List of covers to visualize [default: 10,20,30,40,50]
    -e <ext>, --ext <ext>                   Extension to save figures with [default: svg]
"""
import json
import os
import re
import numpy as np
import pandas as pd
from betanegbinfit.bridge_mixalime import read_dist_from_folder
from negbin_fit.helpers import alleles, make_np_array_path, get_p, init_docopt, \
    make_negative_binom_density, make_line_negative_binom_density, calculate_gof_for_point_fit, \
    ParamsHandler, calculate_overall_gof, check_weights_path, add_BAD_to_path, merge_dfs, read_dfs, get_counts_column, \
    parse_input, parse_files_list, check_states
from negbin_fit.helpers import available_concentrations, available_models,\
    available_dists, available_window_behaviors, available_bnb_models
from negbin_fit.neg_bin_weights_to_df import main as convert_weights
from negbin_fit.visualize import main as visualize
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
    if re.match(r'\d+:\d+:\d+', list_as_string):
        cover_list = list(range(*map(int, list_as_string.split(':'))))
    else:
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


def parse_args(dfs):
    assert dfs is not None
    return merge_dfs([x[1] for x in dfs])


def search_BADs(base_out_path):
    return [float(x[3:]) for x
            in os.listdir(base_out_path)
            if re.match(r'^BAD([1-9]+[.])?[0-9]+$', x)
            and os.path.isdir(os.path.join(base_out_path, x))]


def start_fit():
    schema = Schema({
        '-I': Or(
            Const(lambda x: x == []),
            And(
                Const(lambda x: sum(os.path.exists(y) for y in x),
                      error='Input file(s) should exist'),
                Use(read_dfs, error='Wrong format stats file')
            )
        ),
        '-f': Or(
            Const(lambda x: x is None),
            Use(parse_files_list, error='Error while parsing file -f')
        ),
        '--output': Or(
            Const(lambda x: x is None),
            And(
                Const(check_output, error="No output path exist"),
                Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
            )
        ),
        '--reads-left-tr': And(Use(int),  Const(lambda x: x >= 0), error='Allelic reads threshold must be a non negative integer'),
        '--reads-right-tr': Or(
            And(Use(int),  Const(lambda x: x >= 0)),
            And(Use(str), Const(lambda x: x in ('inf', ))),
            error='Main allelic reads threshold must be a non negative integer',
        ),
        '--reads-right-tr-alt':Or(
            And(Use(int),  Const(lambda x: x >= 0)),
            And(Use(str), Const(lambda x: x in ('inf', 'none'))),
            error='Alternative allelic reads threshold must be a non negative integer',
        ),
        '--states': Or(
            Const(lambda x: x is None),
            Use(
                check_states, error='''Incorrect value for --states.
            Must be "," separated list of numbers or fractions in the form "x/y", each >= 1'''
            )
        ),
        '--cover-list': Use(parse_cover_list, error='Wrong format cover list'),
        '--max-read-count': And(
            Use(int),
            Const(lambda x: x > 0),
            error='Max read count threshold must be a positive integer'
        ),
        '--model': Const(lambda x: x in available_models,
                         error='Model not in ({})'.format(', '.join(available_models))),
        '--ext': Const(lambda x: len(x) > 0),
        '--concentration': Const(lambda x: x in available_concentrations,
                                 error='Concentration mode not in ({})'.format(', '.join(available_concentrations))),
        '--distribution': Const(lambda x: x in available_dists,
                                 error='Distribution not in ({})'.format(', '.join(available_dists))),
        '--window_behavior': Const(lambda x: x in available_window_behaviors,
                                 error='Concentration mode not in ({})'.format(', '.join(available_window_behaviors))),
        '--min_slices': And(Use(int), Const(lambda x: x >= 0),
                            error='Min_slices should be non-negative.'),
        '--window_size': And(Use(int), Const(lambda x: x > 0),
                            error='Min_slices should be positive.'),
        str: bool
    })
    args = init_docopt(__doc__, schema)
    base_out_path = args['--output']
    allele_tr = args['--reads-left-tr']
    model = args['--model']
    to_fit = not args['--no-fit']
    to_visualize = args['--visualize']
    max_fit_count = args['--reads-right-tr']
    if max_fit_count == 'inf':
        max_fit_count = np.inf
    max_fit_count_alt = args['--reads-right-tr-alt']
    if max_fit_count_alt == 'inf':
        max_fit_count_alt = np.inf
    elif max_fit_count_alt == 'none':
        max_fit_count_alt = max_fit_count
    max_read_count = args['--max-read-count']
    dist = args['--distribution']
    window_behavior = args['--window_behavior']
    min_slices = args['--min_slices']
    concentration = args['--concentration']
    window_size = args['--window_size']
    njobs = -1
    stats_dfs = {}
    merged_df = None
    if args['collect']:
        dfs = parse_input(args['-I'], args['-f'])
        _, unique_BADs, merged_df = parse_args(dfs)
        print('{} unique BADs detected'.format(len(unique_BADs)))
    else:
        if args['--states']:
            unique_BADs = args['--states']
        else:
            unique_BADs = search_BADs(base_out_path)
            if len(unique_BADs) > 0:
                print('{} unique BADs detected'.format(len(unique_BADs)))
            else:
                raise ValueError('No BADs found in {}'.format(base_out_path))

    for BAD in sorted(unique_BADs):
        bad_out_path = add_BAD_to_path(base_out_path, BAD)
        if args['collect']:
            print('Collecting stats file for BAD={:.2f} ...'.format(BAD))
            collect_stats_df(merged_df, bad_out_path, BAD)
        else:
            if not os.path.isfile(get_stats_df_path(bad_out_path)):
                raise ValueError('No stats file in {}'.format(bad_out_path))
            stats_df = open_stats_df(bad_out_path)
            stats_dfs[BAD] = stats_df
    if args['collect']:
        exit(0)
    if model not in available_bnb_models:
        line_fit = model == 'NB_AS_Total'
        for BAD in sorted(unique_BADs):
            bad_out_path = add_BAD_to_path(base_out_path, BAD)

            stats_df = stats_dfs[BAD]
            if to_fit:
                d = main(stats_df,
                         out=bad_out_path,
                         BAD=BAD,
                         line_fit=line_fit,
                         allele_tr=allele_tr,
                         max_read_count=max_fit_count)
                if not line_fit:
                    convert_weights(in_df=stats_df,
                                    np_weights_dict=d,
                                    out_path=bad_out_path)
            else:
                try:
                    _, d = check_weights_path(bad_out_path, line_fit=line_fit)
                except Exception:
                    print(__doc__)
                    exit('Error reading weights in {}'.format(bad_out_path))
                    raise
            if to_visualize:
                visualize(
                    stats=stats_df,
                    params=d,
                    model=model,
                    cover_list=args['--cover-list'],
                    max_read_count=max_read_count,
                    out=bad_out_path,
                    BAD=BAD,
                    allele_tr=allele_tr)

    else:
        if to_fit:
            from betanegbinfit import run
            #  FIXME change signature
            #  stats_dfs is a dict where BAD is a key and the corresponding stats_df is a value
            fit_params = run(data=stats_dfs, output_folder=base_out_path,
                             bads=unique_BADs,
                             model=model,
                             window_size=window_size,
                             window_behavior=window_behavior,
                             min_slices=min_slices,
                             estimate_p=False,
                             dist=dist,
                             concentration=concentration,
                             left=allele_tr - 1,
                             max_count=max_fit_count,
                             max_count_alt=max_fit_count_alt,
                             apply_weights=False,
                             n_jobs=njobs)
        else:
            fit_params = read_dist_from_folder(folder=base_out_path)
        if to_visualize:
            for BAD in sorted(unique_BADs):
                bad_out_path = add_BAD_to_path(base_out_path, BAD)
                stats_df = stats_dfs[BAD]
                visualize(
                    stats=stats_df,
                    params=fit_params,
                    model=model,
                    cover_list=args['--cover-list'],
                    image_type=args['--ext'],
                    max_read_count=max_read_count,
                    out=bad_out_path,
                    BAD=BAD,
                    allele_tr=allele_tr)
