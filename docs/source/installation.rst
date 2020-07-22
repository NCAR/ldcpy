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
