.. image:: https://img.shields.io/github/workflow/status/NCAR/ldcpy/CI?logo=github&style=for-the-badge
    :target: https://github.com/NCAR/ldcpy/actions
    :alt: GitHub Workflow CI Status

.. image:: https://img.shields.io/github/workflow/status/NCAR/ldcpy/code-style?label=Code%20Style&style=for-the-badge
    :target: https://github.com/NCAR/ldcpy/actions
    :alt: GitHub Workflow Code Style Status

.. image:: https://img.shields.io/codecov/c/github/NCAR/ldcpy.svg?style=for-the-badge
    :target: https://codecov.io/gh/NCAR/ldcpy

Lossy Data Compression for Python
=================================

Development
------------

For a development install, do the following in the repository directory:

.. code-block:: bash

    conda env update -f ci/environment.yml
    conda activate ldcpy_env
    python -m pip install -e .
