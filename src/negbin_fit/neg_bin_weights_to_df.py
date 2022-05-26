import pandas as pd
from negbin_fit.helpers import alleles, read_weights, get_nb_weight_path


def main(out_path, BAD, in_file=None, in_df=None, np_weights_path=None,
         np_weights_dict=None, line_fit=False):
    column_names = ['r', 'w', 'status', 'gof']
    if in_file is not None:
        counts_df = pd.read_table(in_file)
    elif in_df is not None:
        counts_df = in_df
    else:
        raise AssertionError('No input file provided')
    all_df_list = []
    for main_allele in alleles:
        np_weights = read_weights(
            np_weights_path=np_weights_path,
            np_weights_dict=np_weights_dict,
            line_fit=line_fit,
            allele=main_allele)
        df = pd.DataFrame(columns=column_names)
        for i in range(len(column_names)):
            df[column_names[i]] = np_weights[:, i]
        counts = []
        for fix_c in df.index:
            counts.append(counts_df[counts_df[alleles[main_allele]] == fix_c]['counts'].sum())
        df['allele_reads'] = counts
        df.to_csv(get_nb_weight_path(out_path, main_allele, BAD),
                  index=False, sep='\t')
        df['allele'] = main_allele
        all_df_list.append(df)
    all_df = pd.concat(all_df_list)
    all_df['BAD'] = BAD
    all_df.to_csv(get_nb_weight_path(out_path, main_allele, BAD, result=True),
                  index=False, sep='\t')

