#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from betanegbinfit import ModelMixture, ModelMixtures, ModelLine, ModelWindow
from betanegbinfit.models import ModelLine_
from functools import partial
from glob import glob
import datatable as dt
import logging
import lzma
import gzip
import bz2
import os
import re


openers = {'lzma': lzma.open,
           'gzip': gzip.open,
           'bz2': bz2.open,
           'raw': open}


def get_init_file(path: str):
    folder, name = os.path.split(path)
    for file in os.listdir(folder if folder else None):
        if file.startswith(f'{name}.init.') and file.endswith(tuple(openers.keys())) and os.path.isfile(file):
            return os.path.join(folder, file)

def get_init_files(path: str):
    files = list()
    folder, name = os.path.split(path)
    ptrn = re.compile(name + r'.init.\d+.\w+')
    for file in os.listdir(folder if folder else None):
        m = ptrn.fullmatch(file)
        if m is not None and (m.start() == 0) and (m.end() == len(file)):
            files.append(file)
    return [os.path.join(folder, x) for x in sorted(files, key=lambda x: int(x.split('.')[-2]))]

def get_model_creator(**kwargs):
    name = kwargs['name']
    inst_params = {v: kwargs[v] for v in ('bad',  'left', 'dist', 'estimate_p')}
    if name == 'line':
        inst_params.update({v: kwargs[v] for v in ('left_k', 'start_est', 'apply_weights')})
        m = ModelLine
    elif name == 'window':
        inst_params.update({v: kwargs[v] for v in ('window_size', 'left_k', 'window_behavior', 'min_slices',
                                                   'adjust_line', 'start_est', 'apply_weights', 'regul_alpha',
                                                   'regul_n', 'regul_slice', 'regul_prior')})
        m = ModelWindow
    elif name == 'slices':
        m = ModelMixtures
    elif name == 'slice':
        m = ModelMixture
    elif name == 'line_diff':
        inst_params.update({v: kwargs[v] for v in ('left_k', 'start_est', 'apply_weights')})
        m = ModelLine_
    else:
        raise Exception(f'Unknown model name {name}.')
    return partial(m, **inst_params)
    
def dictify_params(d: dict, field='ests') -> dict:
    return {n: v for n, v in zip(d['names'], d[field])}

def parse_filenames(files: list) -> list:
    if type(files) is str:
        files = [files]
    res = list()
    for file in files:
        if file.startswith('m:'):
            file = file[len('m:'):]
            for file in glob(file):
                res.append(file)
        elif os.path.isdir(file):
            res.extend(filter(lambda x: os.path.isfile(x) and not x.endswith('.tbi'), 
                              (os.path.join(file, f) for f in os.listdir(file))))
        elif os.path.isfile(file):
            if not file.endswith(('.gz', '.vcf', '.bam')):
                folder, _ = os.path.split(file)
                df = dt.fread(file, max_nrows=1)
                if (df.shape[1] == 1) and os.path.isfile(os.path.join(file, df[0, 0]) if folder else df[0, 0]):
                    df = dt.fread(file)
                    for i in range(df.shape[0]):
                        file = os.path.join(folder, df[i, 0]) if folder else df[i, 0]
                        if not os.path.isfile(file):
                            logging.error(f'File {file} not found.')
                        else:
                            res.append(file)
                else:
                    res.append(file)
            else:
                res.append(file)
        else:
            logging.error(f'File {file} not found.')
    res = sorted(res)
    return res
