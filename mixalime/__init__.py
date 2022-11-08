__version__ = '2.2.9'
import importlib

__min_reqs__ = [
            'pip>=22.3',
            'typer>=0.6.1',
            'numpy>=1.23.4',
            'jax>=0.3.23',
            'matplotlib>=3.5.1',
            'pandas>=1.4.1',
            'scipy>=1.9.3',
            'statsmodels>=0.13.2',
            'betanegbinfit>=1.0.4',
            'datatable>=1.0.0',
            'dill>=0.3.6',
            'rich>=12.6.0',
            'portion>=2.3.0',
            'pysam>=0.19.1'
           ]

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def check_packages():
    for req in __min_reqs__:
        module, ver = req.split('>=')
        ver = versiontuple(ver)
        try:
            v = versiontuple(importlib.import_module(module).__version__)
        except AttributeError:
            continue
        if v < ver:
            raise ImportError(f'Version of the {module} package should be at least {ver} (found: {v}).')
