from mixalime import __version__
from setuptools import setup, find_packages
import os

with open('README.md', 'r', encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name='mixalime',
    version=__version__,
    packages=find_packages(),
    include_package_data=True,
    package_data={'': ['data/*']},
    description='Identification of allele-specific events in sequencing experiments.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    entry_points={
        'console_scripts': [
            'mixalime = mixalime.main:main',
        ],
    },
    author='Georgy Meshcheryakov',
    author_email='iam@georgy.top',
    install_requires=[
        'pip>=21.1.3',
        'typer>=0.6.1',
        'numpy>=1.21.5',
        'jax>=0.3.23',
        'matplotlib>=3.5.1',
        'pandas>=1.4.1',
        'matplotlib>=3.2.1',
        'scipy>=1.8.1',
        'statsmodels>=0.13.2',
        'betanegbinfit>=1.0.0',
        'datatable>=1.0.0',
        'dill>=0.3.4',
        'rich>=12.6.0',
        'portion>=2.3.0',
        'pysam>=0.19.1'
    ],
    python_requires='>=3.6, <3.10',
    url="https://github.com/autosome-ru/mixalime",
    classifiers=[
              "Programming Language :: Python :: 3.7",
	      "Programming Language :: Python :: 3.8",
	      "Programming Language :: Python :: 3.9",
	      "Development Status :: 5 - Production/Stable",
	      "Topic :: Scientific/Engineering",
              "Operating System :: OS Independent"]

)