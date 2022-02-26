import numpy as np
import os

import pandas as pd
from matplotlib import pyplot as plt, ticker
import seaborn as sns
from negbin_fit.helpers import alleles, make_out_path, get_pvalue_file_path


def main(dfs, BADs, out, ext='svg'):
    for df in dfs:
        df_name, result_df = df
        out = make_out_path(out, 'visualizations')
        for BAD in sorted(BADs):
            bad_df = result_df[result_df['BAD'] == BAD]
            if len(result_df.index) == 0:
                print('No p-values for BAD={}'.format(BAD))
                continue
            print('Visualizing p-value for {}, BAD={}'.format(df_name, BAD))
            vis_p_value_bias(bad_df=bad_df,
                             df_name=df_name,
                             BAD=BAD,
                             out=out,
                             ext=ext)
            vis_p_value_dist(bad_df=bad_df,
                             df_name=df_name,
                             BAD=BAD,
                             ext=ext,
                             out=out)


def get_field(allele):
    return 'PVAL_' + allele.upper()


def filter_p(left_b, right_b, df, field):
    return len(df[(left_b < df[field])
                  & (df[field] <= right_b)].index)


def get_y(p_list, field, x):
    return [filter_p(1 / 10 ** (k + 1), 1 / 10 ** k, p_list, field) for k in x]


def vis_p_value_bias(bad_df, df_name, BAD, out, ext):
    x = np.array(range(0, 18))
    fig, ax = plt.subplots(figsize=(10, 8))

    for color, allele in zip(['C0', 'C1'], alleles):
        field = get_field(allele)
        p_list = bad_df[bad_df[field] != 1.0]
        sns.barplot(x=x, y=get_y(p_list, field, x),
                    label=allele, ax=ax, color=color, alpha=0.5)

    plt.grid(True)

    plt.title('ref-alt p_value on BAD={:.2f}'.format(BAD))
    plt.xlabel('x: x+1 >-log10 p_value >= x')
    plt.ylabel('snp count')
    plt.savefig(os.path.join(out, '{}.pval_bias.BAD{:.2f}.{}'.format(df_name, BAD, ext)))
    plt.close(fig)


def vis_p_value_dist(bad_df, df_name, BAD, out, ext='svg'):
    fig, ax = plt.subplots(figsize=(10, 8))
    x = np.linspace(0, 1, 50)
    sns.barplot(x=x[:-1],
                y=[sum([filter_p(x[i], x[i + 1], bad_df, get_field(allele))
                       for allele in alleles])
                   for i in range(len(x) - 1)],
                label='sum', ax=ax, color='grey')

    ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0, len(x), 5)))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(['{:.1f}'.format(f) for f in x[::5]]))
    ax.tick_params(axis="x", rotation=90)

    plt.grid(True)
    plt.legend()
    plt.title('p-value distribution for BAD={:.2f}'.format(BAD))
    plt.xlabel('p-value')
    plt.ylabel('snp count')

    plt.savefig(os.path.join(out, '{}.pval_dist.BAD{:.2f}.{}'.format(df_name, BAD, ext)))
    plt.close(fig)
