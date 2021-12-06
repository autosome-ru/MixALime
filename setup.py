from setuptools import setup, find_packages
from io import open
import os

version_module = {}
dir_name = os.path.dirname(__file__)
with open(os.path.join(dir_name, "src", "negbin_fit", "version.py")) as fp:
    exec(fp.read(), version_module)
    __version__ = version_module['__version__']

with open(os.path.join(dir_name, "README.md"), encoding='utf8') as fh:
    long_description = fh.read()

setup(
    name='negbin_fit',
    version=__version__,
    packages=find_packages('src'),
    include_package_data=True,
    # package_data={'babachi': ['tests/*.tsv']},
    long_description=long_description,
    entry_points={
        'console_scripts': [
            'negbin_fit = negbin_fit.fit_nb:start_fit',
            'cover_fit = negbin_fit.fit_cover:main',
            'calc_pval = calculate_pvalue.calculate_pvalue:main'
        ],
    },
    author="Sergey Abramov, Alexandr Boytsov",
    author_email='aswq22013@gmail.com',
    package_dir={'': 'src'},
    install_requires=[
        'docopt>=0.6.2',
        'numpy>=1.18.0',
        'schema>=0.7.2',
        'contextlib2>=0.5.5',
        'pandas>=1.0.4',
        'matplotlib>=3.2.1',
        'seaborn>=0.10.1',
        'scipy>=1.5.2',
        'tqdm>=4.62.3',
        'statsmodels>=0.12.2'
    ],
    extras_require={
        'dev': ['wheel', 'twine', 'setuptools_scm'],
    },
    python_requires='>=3.6',
    url="https://github.com/wish1/neg_bin_fit",
)
