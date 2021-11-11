import os
import numpy as np
import json


alleles = {'ref': 'alt', 'alt': 'ref'}


def make_np_array_path(out, allele, line_fit=False):
    return os.path.join(out, allele + '.' + ('npy' if not line_fit else 'json'))


def get_nb_weight_path(out, allele):
    return os.path.join(out, 'NBweights_{}.tsv'.format(allele))


def read_weights(allele, np_weights_path=None, np_weights_dict=None, line_fit=False):
    if np_weights_path:
        path = make_np_array_path(np_weights_path, allele, line_fit=line_fit)
        if not line_fit:
            np_weights = np.load(path)
        else:
            with open(path) as r:
                np_weights = json.load(r)
    elif np_weights_dict:
        np_weights = np_weights_dict[allele]
    else:
        raise AssertionError('No numpy fits provided')
    return np_weights


def get_p(BAD):
    return 1 / (BAD + 1)
