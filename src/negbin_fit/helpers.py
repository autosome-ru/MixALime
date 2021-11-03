import os
import numpy as np

alleles = {'ref': 'alt', 'alt': 'ref'}


def make_np_array_path(out, allele):
    return os.path.join(out, allele + '.npy')


def get_nb_weight_path(out, allele):
    return os.path.join(out, 'NBweights_{}.tsv'.format(allele))


def read_weights(allele, np_weights_path=None, np_weights_dict=None):
    if np_weights_path:
        np_weights = np.load(make_np_array_path(np_weights_path, allele))
    elif np_weights_dict:
        np_weights = np_weights_dict[allele]
    else:
        raise AssertionError('No numpy fits provided')
    return np_weights


def get_p(BAD):
    return 1 / (BAD + 1)
