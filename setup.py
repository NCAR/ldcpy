#!/usr/bin/env python

"""The setup script."""

import os
import sys
from os.path import exists

from setuptools import find_packages, setup
from setuptools.command.install import install

VERSION = 'v0.10'

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

# with open('requirements-dev.txt') as f:
#    dev_requires = f.read().strip().split('\n')

if exists('README.rst'):
    with open('README.rst') as f:
        long_description = f.read()
else:
    long_description = ''

CLASSIFIERS = [
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
]


class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = 'verify that the git tag matches our version'

    def run(self):
        tag = os.getenv('CIRCLE_TAG')

        if tag != VERSION:
            info = 'Git tag: {0} does not match the version of this app: {1}'.format(tag, VERSION)
            sys.exit(info)


setup(
    name='ldcpy',
    version=VERSION,
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
    #    extras_require={'dev': dev_requires},
    license='Apache 2.0',
    zip_safe=False,
    keywords='compression, xarray',
    use_scm_version={'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'},
    setup_requires=['setuptools_scm', 'setuptools>=30.3.0'],
    cmdclass={'verify': VerifyVersionCommand},
)
