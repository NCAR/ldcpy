#!/usr/bin/env python3

"""The setup script."""

from setuptools import find_packages, setup

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

with open('README.rst') as f:
    long_description = f.read()


CLASSIFIERS = [
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering',
]

setup(
    name='ldcpy',
    description='A library for lossy compression of netCDF files using xarray',
    long_description=long_description,
    python_requires='>=3.6',
    maintainer='Alex Pinard',
    maintainer_email='apinard@mines.edu',
    classifiers=CLASSIFIERS,
    url='https://ldcpy.readthedocs.io',
    project_urls={
        'Documentation': 'https://ldcpy.readthedocs.io',
        'Source': 'https://github.com/NCAR/ldcpy',
        'Tracker': 'https://github.com/NCAR/ldcpy/issues',
    },
    packages=find_packages(exclude=('tests',)),
    package_dir={'ldcpy': 'ldcpy'},
    include_package_data=True,
    install_requires=install_requires,
    license='Apache 2.0',
    zip_safe=False,
    keywords='compression, xarray',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
)
