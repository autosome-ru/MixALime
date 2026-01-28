# -*- coding: utf-8 -*-
__version__ = '0.26.0'
import importlib


__min_reqs__ = [
            'pip>=24.0',
            'typer>=0.13',
            'numpy>=2.1',
            'jax>=0.8',
            'jaxlib>=0.8',
            'matplotlib>=3.5',
            'pandas>=2.2',
            'scipy>=1.14',
            'statsmodels>=0.14',
            'datatable>=1.0.0' ,
            'dill>=0.3.9',
            'rich>=12.6.0',
            'tqdm>=4.0',
            'scikit-learn>=1.6',
            'tables>=3.10',
            'sympy>=1.12',
            'seaborn>=0.12'
           ]

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def check_packages():
    for req in __min_reqs__:
        try:
            module, ver = req.split(' @').split('>=')
            ver = versiontuple(ver)
            v = versiontuple(importlib.import_module(module).__version__)
        except (AttributeError, ValueError):
            continue
        if v < ver:
            raise ImportError(f'Version of the {module} package should be at least {ver} (found: {v}).')
