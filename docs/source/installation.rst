============
Installation
============


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
    conda activate ldcpy

Install cartopy (must install using conda):

.. code-block:: bash

    conda install cartopy

Then install ldcpy:

.. code-block:: bash

    pip install ldcpy

Accessing the tutorial (for users)
__________________________________

If you want access to the tutorial notebook, clone the repository (this will create a local repository in the current directory):

.. code-block:: bash

    git clone https://github.com/NCAR/ldcpy.git

Start by enabling Hinterland for code completion in Jupyter Notebook and then opening the tutorial notebook:

.. code-block:: bash

    jupyter nbextension enable hinterland/hinterland
    jupyter notebook


The tutorial notebook can be found in docs/source/notebooks/SampleNotebook.ipynb, feel free to gather your own metrics or create your own plots in this notebook!


Installation for Developers
___________________________

First, clone the repository and cd into the root of the repository:


.. code-block:: bash

    git clone https://github.com/NCAR/ldcpy.git
    cd ldcpy

For a development install, do the following in the ldcpy repository directory:

.. code-block:: bash

    conda env update -f environment-dev.yml
    conda activate ldcpy
    python -m pip install -e .

This code block enables optional extensions for code completion, code hinting and minimizing tracebacks in Jupyter. Then start the jupyter notebook server in your browser (at localhost:8888):

.. code-block:: bash

    jupyter nbextension enable hinterland/hinterland
    jupyter nbextension enable skip-traceback/main

    conda activate ldcpy
    jupyter notebook

For viewing changes to documentation in the repo, do the following:

.. code-block:: bash

    pip install -r docs/requirements.txt
    sphinx-reload docs/

This starts and opens a local version of the documentation in your browser (at localhost:5500/index.html) and keeps it up to date with any changes made. Note that changes to docstrings in the code will not trigger an update, only changes to the .rst files in the docs/ folder.

Before committing changes to the code, run the tests from the project root directory to ensure they are passing.

.. code-block:: bash

    pytest -n 4

pre-commit should automatically run black, flake8, and isort to enforce style guidelines. If changes are made, the first commit will fail and you will need to stage the changes that have been made before committing again. If, for some reason, pre-commit fails to make changes to your files, you should be able to run the following to clean the files manually:

.. code-block:: bash

    black --skip-string-normalization --line-length=100 .
    flake8 .
    isort .
