============
Installation
============


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


Installation for Developers
------------

For a development install, do the following in the ldcpy repository directory:

.. code-block:: bash

    conda env update -f environment_dev.yml
    conda activate ldcpy
    python -m pip install -e .

Optional extensions for code completion and minimizing tracebacks:

.. code-block:: bash
    jupyter nbextension enable hinterland/hinterland
    jupyter nbextension enable skip-traceback/main

For viewing changes to documentation in the repo, do the following:

.. code-block:: bash
    cd docs/
    sphinx reload .

Then start a local version of the documentation and keep it up to date with any changes made.

Before committing your code, run the tests from the project root directory to ensure they are passing.

.. code-block:: bash
    pytest

 pre-commit should automatically run blake, flake8, and isort to enforce style guidelines. If changes are made, the first commit will fail and you will need to stage the changes that have been made before committing again. If, for some reason, pre-commit fails to make changes to your files, you should be able to run the following to clean the files:

.. code-block:: bash
    black --skip-string-normalization --line-length=100 .
    flake8 .
    isort .

Documentation and usage examples are available `here <http://ldcpy.readthedocs.io>`_.
