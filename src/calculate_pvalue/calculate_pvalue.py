"""
Usage:
    calc_pval (<file>...) [-O <dir> |--output <dir>] (-w <dir> | --weights <dir>)
    cover_fit -h | --help


Arguments:
    <file>            Path to input file in tsv format
    <dir>             Directory name

Options:
    -h, --help                              Show help.
    -O <path>, --output <path>              Output directory for obtained fits. [default: ./]
    -w <dir>, --weights <dir>               Directory with fitted weights
"""

import os
import pandas as pd
from negbin_fit.helpers import init_docopt, alleles, check_weights_path
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


def filter_df(merged_df, key):
    return merged_df[merged_df['key'] == key]


def merge_dfs(dfs):
    merged_df = pd.concat(dfs, names=['key'], ignore_index=True)
    return merged_df['key'].unique(), merged_df


def get_key(row):
    return row['ID']


def read_df(filename):
    try:
        df = pd.read_table(filename)
        df['key'] = get_key(df)
        assert df.columns
    except Exception:
        raise AssertionError
    return os.path.splitext(os.path.basename(df)), df


def get_posterior_weights(merged_df, unique_snps, weights):
    result = {}
    for snp in unique_snps:
        result[snp] = {'ref': {}, 'alt': {}}
        filtered_df = filter_df(merged_df, snp)
        for main_allele in alleles:
            prior_weight = weights[alleles[main_allele]]
            main_counts = filtered_df[main_allele + '_counts'].to_list()
            fixed_counts = filtered_df[alleles[main_allele] + '_counts'].to_list()
    return result


def start_process(dfs, out_path, fit_weights):
    unique_snps, merged_df = merge_dfs([x[1] for x in dfs])
    weights = get_posterior_weights(merged_df, unique_snps, fit_weights)
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
        '--weights': And(
            Const(os.path.exists, error='Weights dir should exist'),
            Use(check_weights_path, error='Wrong formatted weights')
        ),
        '--output': And(
            Const(os.path.exists),
            Const(lambda x: os.access(x, os.W_OK), error='No write permissions')
        )
    })
    args = init_docopt(__doc__, schema)
    print('Here we go again', args)
    start_process(args['<file>'], args['--output'], args['--weights'])
