============
Installation
============


ldcpy can be set up for development using the following commands (in the ldcpy repo):

.. code-block:: bash

        conda env create --file environment-dev.yml

For usage examples, see Jupyter notebooks. To activate the environment and start Jupyter, do the following:

.. code-block:: bash

        conda activate ldcpy_dev
        jupyter nbextension enable hinterland/hinterland
        jupyter nbextension enable skip-traceback/main
        jupyter notebook

The sample notebook can be found in docs/source/notebooks/SampleNotebook.ipynb, feel free to gather your own metrics or create your own plots in this notebook!
