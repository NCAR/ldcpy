.. image:: https://img.shields.io/github/workflow/status/NCAR/ldcpy/CI?logo=github&style=for-the-badge
    :target: https://github.com/NCAR/ldcpy/actions
    :alt: GitHub Workflow CI Status

.. image:: https://img.shields.io/circleci/project/github/NCAR/ldcpy/master.svg?style=for-the-badge&logo=circleci
    :target: https://circleci.com/gh/NCAR/ldcpy/tree/master

.. image:: https://img.shields.io/github/workflow/status/NCAR/ldcpy/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/NCAR/ldcpy/actions
    :alt: GitHub Workflow Code Style Status

.. image:: https://img.shields.io/codecov/c/github/NCAR/ldcpy.svg?style=for-the-badge
    :target: https://codecov.io/gh/NCAR/ldcpy

.. image:: https://img.shields.io/readthedocs/ldcpy/latest.svg?style=for-the-badge
    :target: https://ldcpy.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Lossy Data Compression for Python
=================================

ldcpy is a utility for gathering and plotting metrics from NetCDF files using the Pangeo stack.

Installation for Users
___________

Activate the base cartopy environment:

.. code-block:: bash

    conda activate

Install cartopy (must install using conda):

.. code-block:: bash

    conda install cartopy

Then install ldcpy:

.. code-block:: bash

    pip install ldcpy

Start by enabling Hinterland for code completion in Jupyter Notebook and then opening the tutorial notebook:

.. code-block:: bash

    jupyter nbextension enable hinterland/hinterland
    jupyter notebook

The tutorial notebook can be found in docs/source/notebooks/SampleNotebook.ipynb, feel free to gather your own metrics or create your own plots in this notebook!
