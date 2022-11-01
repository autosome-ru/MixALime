from mixalime import __version__, __min_reqs__
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
    install_requires=__min_reqs__,
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