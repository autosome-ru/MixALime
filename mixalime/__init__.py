__version__ = '2.14.11'
import importlib
import sys

ge_10 = sys.version_info[1] >= 10

__min_reqs__ = [
            'pip>=23.0',
            'typer>=0.6.1',
            'numpy>=1.23.4',
            'jax>=0.4.4',
            'jaxlib>=0.4.4',
            'matplotlib>=3.5.1',
            'pandas>=1.4.1',
            'scipy>=1.10.1',
            'statsmodels>=0.13.2',
            'betanegbinfit>=1.9.5',
            'datatable @ git+https://github.com/h2oai/datatable.git' if ge_10 else 'datatable>=1.0.0' ,
            'dill>=0.3.6',
            'rich>=12.6.0',
            'portion>=2.3.0',
            'pysam>=0.19.1'
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
