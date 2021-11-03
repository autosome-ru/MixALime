import pandas as pd
from negbin_fit.helpers import alleles, read_weights, get_nb_weight_path


def main(out_path, in_file=None, in_df=None, np_weights_path=None,
         np_weights_dict=None):
    column_names = ['r', 'w', 'status', 'gof']
    if in_file is not None:
        counts_df = pd.read_table(in_file)
    elif in_df is not None:
        counts_df = in_df
    else:
        raise AssertionError('No input file provided')
    for allele in alleles:
        np_weights = read_weights(
            np_weights_path=np_weights_path,
            np_weights_dict=np_weights_dict,
            allele=allele)
        df = pd.DataFrame(columns=column_names)
        for i in range(len(column_names)):
            df[column_names[i]] = np_weights[:, i]
        counts = []
        for fix_c in df.index:
            counts.append(counts_df[counts_df[allele] == fix_c]['counts'].sum())
        df['allele_reads'] = counts
        df.to_csv(get_nb_weight_path(out_path, allele),
                  index=False, sep='\t')
