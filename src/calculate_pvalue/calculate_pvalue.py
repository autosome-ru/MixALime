"""
Usage:
    calc_pval (<file>...) [-O <dir> |--output <dir>]
    cover_fit -h | --help


Arguments:
    <file>            Path to input file in tsv format
    <dir>             Directory for fitted weights

Options:
    -h, --help                              Show help.
    -O <path>, --output <path>              Output directory for obtained fits. [default: ./]
"""

import os
import pandas as pd
from negbin_fit.helpers import init_docopt
from schema import Schema, And, Const, Use


def calculate_pval(row, row_weights):
    return 0, 0, 0, 0


def process_df(row, weights):
    p_ref, p_alt, es_ref, es_alt = calculate_pval(row, weights[get_key(row)])
    row['pval_ref'] = p_ref
    row['pval_alt'] = p_alt
    row['es_ref'] = es_ref
    row['es_alt'] = es_alt
    return row


def merge_dfs(dfs):
    return pd.concat(dfs)


def get_key(row):
    pass


def read_df(filename):
    try:
        df = pd.read_table(filename)
        assert df.columns
    except Exception:
        raise AssertionError
    return os.path.splitext(os.path.basename(df)), df


def get_posterior_weights(merged_df):
    return {}


def start_process(dfs, out_path):
    merged_df = merge_dfs([x[1] for x in dfs])
    weights = get_posterior_weights(merged_df)
    for df_name, df in dfs:
        df = df.apply(lambda x: process_df(x, weights), axis=1)
        df.to_csv(os.path.join(out_path, df_name + '.pvalue_table'),
                  sep='\t', index=False)


def main():
    schema = Schema({
        '<file>': And(
            Const(os.path.exists, error='Input file(s) should exist'),
            Use(read_df, error='Wrong format stats file')
        ),
        '--output': And(
            Const(os.path.exists),
            Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
        )
    })
    args = init_docopt(__doc__, schema)
    print('Here we go again', args)
    start_process(args['<file>'], args['--output'])
