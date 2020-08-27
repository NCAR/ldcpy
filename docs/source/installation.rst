============
Installation
============


Installation using Conda (recommended)
______________________________________

Ensure conda is up to date and create a clean Python (3.6+) environment:

.. code-block:: bash

    conda update conda
    conda create --name ldcpy python=3.8
    conda activate ldcpy

Now install ldcpy:

.. code-block:: bash

    conda install -c conda-forge ldcpy

Alternative Installation
________________________

Ensure pip is up to date, and your version of python is at least 3.6:

.. code-block:: bash

    pip install --upgrade pip
    python --version

Install cartopy using the instructions provided at https://scitools.org.uk/cartopy/docs/latest/installing.html.

Then install ldcpy:

.. code-block:: bash

    pip install ldcpy


Accessing the tutorial (for users)
__________________________________

If you want access to the tutorial notebook, clone the repository (this will create a local repository in the current directory):

.. code-block:: bash

    git clone https://github.com/NCAR/ldcpy.git

Start by activating the ldcpy environment, enabling Hinterland for code completion in Jupyter Notebook and then starting the notebook server:

.. code-block:: bash

    conda activate ldcpy
    conda install -c conda-forge jupyter_contrib_nbextensions
    jupyter contrib nbextension install --user
    jupyter nbextension enable hinterland/hinterland
    jupyter notebook


The tutorial notebook can be found in docs/source/notebooks/TutorialNotebook.ipynb, feel free to gather your own metrics or create your own plots in this notebook!
