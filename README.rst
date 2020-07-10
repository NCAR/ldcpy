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
______________________

Ensure conda and pip are up to date, and your version of python is at least 3.6:

.. code-block:: bash

    conda update conda
    pip install --upgrade pip
    python --version

Create a clean conda environment:

.. code-block:: bash

    conda create --name ldcpy python
    conda activate myenv

Install cartopy (must install using conda):

.. code-block:: bash

    conda install cartopy

Then install ldcpy:

.. code-block:: bash

    pip install ldcpy

Accessing the tutorial
______________________

If you want access to the tutorial notebook, clone the repository (this will create a local repository in the current directory):

.. code-block:: bash

    git clone https://github.com/NCAR/ldcpy.git

Start by enabling Hinterland for code completion and code hinting in Jupyter Notebook and then opening the tutorial notebook:

.. code-block:: bash

    jupyter nbextension enable hinterland/hinterland
    jupyter notebook


The tutorial notebook can be found in docs/source/notebooks/SampleNotebook.ipynb, feel free to gather your own metrics or create your own plots in this notebook!
