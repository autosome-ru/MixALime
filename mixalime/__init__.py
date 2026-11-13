__version__ = '2.30.0'
from importlib import metadata
import re


__min_reqs__ = [
            'pip>=23.0',
            'typer>=0.6.1',
            'numpy>=1.21.0',
            'jax<0.9',
            'jaxlib<0.9',
            'matplotlib>=3.5.1',
            'pandas>=1.4.1',
            'scipy<1.16',
            'statsmodels>=0.13.2',
            'jacobin>=1.3.1',
            'polars>=1.0.0',
            'dill>=0.3.7',
            'rich>=12.6.0',
            'pysam>=0.22.0'
           ]

def versiontuple(v):
    """Leading numeric components of a version string, (1, 2, 0) for '1.2.0rc1'."""
    res = list()
    for t in v.split('.'):
        m = re.match(r'\d+', t)
        if not m:
            break
        res.append(int(m.group()))
    return tuple(res)

def check_packages():
    """Raises an ImportError if an installed package does not satisfy its bound in __min_reqs__."""
    for req in __min_reqs__:
        m = re.fullmatch(r'\s*([\w.-]+)\s*(>=|<)\s*([\w.]+)\s*', req)
        if not m:
            continue
        module, op, bound = m.groups()
        if module == 'pip':
            continue
        try:
            found = metadata.version(module)
        except metadata.PackageNotFoundError:
            raise ImportError(f'The package {module} is required ({req}) and is not installed.')
        v, ver = versiontuple(found), versiontuple(bound)
        # Padded with zeros to one length, as (1, 3) sorts before (1, 3, 0).
        n = max(len(v), len(ver))
        v, ver = v + (0,) * (n - len(v)), ver + (0,) * (n - len(ver))
        if (op == '>=' and v < ver) or (op == '<' and v >= ver):
            raise ImportError(f'The package {module} should satisfy {req} (found: {found}).')
