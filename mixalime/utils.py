#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from betanegbinfit import ModelMixture, ModelMixtures, ModelLine, ModelWindow, ModelWindowRec
from betanegbinfit.models import ModelLine_
from collections import defaultdict
from functools import partial
from fnmatch import fnmatch
from glob import glob
from math import ceil
import datatable as dt
import numpy as np
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
        if file.startswith(f'{name}.init.') and file.endswith(tuple(openers.keys())) and os.path.isfile(os.path.join(folder, file)):
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

def dictify_fix(s: str):
    if type(s) is not str:
        return s
    s = s.replace(' ', '').strip()
    return {a: float(b) for t in s.split(';') if t for a, b in [t.split('=')]}     

def get_model_creator(**kwargs):
    name = kwargs['name']
    inst_params = {v: kwargs[v] for v in ('bad',  'left', 'dist', 'estimate_p', 'fix_params', 'r_transform')}
    if name == 'line':
        inst_params.update({v: kwargs[v] for v in ('left_k', 'start_est', 'apply_weights')})
        m = ModelLine
    elif name == 'window':
        inst_params.update({v: kwargs[v] for v in ('window_size', 'left_k', 'window_behavior', 'min_slices',
                                                   'adjust_line', 'start_est', 'apply_weights', 'regul_alpha',
                                                   'regul_n', 'regul_slice', 'regul_prior',
                                                   'symmetrify')})
        m = ModelWindowRec if 'MCNB' in inst_params['dist'] else ModelWindow
    elif name == 'slices':
        m = ModelMixtures
    elif name == 'slice':
        m = ModelMixture
    elif name == 'line_diff':
        inst_params.update({v: kwargs[v] for v in ('left_k', 'start_est', 'apply_weights')})
        m = ModelLine_
    else:
        raise Exception(f'Unknown model name {name}.')
    inst_params['fix_params'] = dictify_fix(inst_params['fix_params'])
    return partial(m, **inst_params)

def scorefiles_qc(init) -> tuple:
    covers = defaultdict(lambda: defaultdict(int))
    biases = defaultdict(lambda: defaultdict(list))
    for _, its in init['snvs'].items():
        its = its[1:]
        for it in its:
            ind, ref, alt = it[:3]
            bad = it[-1]
            covers[ind][bad] += ref + alt
            covers[ind][None] += ref + alt
            biases[ind][bad].append(ref > alt)
            biases[ind][None].append(ref > alt)
    for ind, bads in biases.items():
        for bad in bads:
            biases[ind][bad] = np.mean(biases[ind][bad])
    return covers, biases

def dictify_params(d: dict, field='ests') -> dict:
    return {n: v for n, v in zip(d['names'], d[field])}

def select_filenames(patterns: list, files) -> list:
    res = list()
    for pattern in patterns:
        subres = list()
        if pattern.startswith('m:'):
            t = pattern[2:]
            subres.extend(filter(lambda x: fnmatch(x, t), files))
        elif pattern in files:
            subres.append(pattern)
        else:
            if not pattern.endswith('/'):
                t = pattern + '/'
            else:
                t = pattern
            subres.extend(filter(lambda x: x.startswith(t), files))
        if not subres:
            subres.extend(filter(lambda x: x in files, parse_filenames(pattern, ignore_errors=True)))
            if not subres:
                logging.error(f'No files agree with the pattern {pattern}.')
        res.extend(subres)
    return res
            

def parse_filenames(files: list, files_list=None, ignore_errors=False) -> list:    
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
            if not file.endswith(('.gz', '.vcf', '.bam', '.bcf')):
                folder, _ = os.path.split(file)
                header = True
                df = dt.fread(file, max_nrows=1, header=header)
                if df.ncols == 1:
                    header = False
                    df = dt.fread(file, max_nrows=1, header=header)
                r = str(df[0, 0])
                t, _ = os.path.split(r)
                if (df.shape[1] == 1) and ((os.path.isabs(t) and os.path.isfile(r)) or os.path.isfile(os.path.join(folder, r))):
                    df = dt.fread(file, header=header)
                    for i in range(df.shape[0]):
                        r = str(df[i, 0])
                        file = os.path.join(folder, df[i, 0]) if folder and not os.path.isabs(r) else df[i, 0]
                        if not os.path.isfile(file) and not ignore_errors:
                            logging.error(f'File {file} not found.')
                        else:
                            res.append(file)
                else:
                    res.append(file)
            else:
                res.append(file)
        elif not ignore_errors:
            logging.error(f'File {file} not found.')
    res = sorted(res)
    return res
