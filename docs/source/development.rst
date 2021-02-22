===========
Development
===========

Installation for Developers
___________________________

First, clone the repository and cd into the root of the repository:

.. code-block:: bash

    git clone https://github.com/NCAR/ldcpy.git
    cd ldcpy

For a development install, do the following in the ldcpy repository directory:

.. code-block:: bash

    conda env update -f environment.yml
    conda activate ldcpy
    python -m pip install -e .

Then install the pre-commit script and git hooks for code style checking:

.. code-block:: bash

    pre-commit install

This code block enables optional extensions for code completion, code hinting and minimizing tracebacks in Jupyter. Then start the jupyter notebook server in your browser (at localhost:8888):

.. code-block:: bash

    jupyter nbextension enable hinterland/hinterland
    jupyter nbextension enable skip-traceback/main

    conda activate ldcpy
    jupyter notebook

Instructions and Tips for Contributing
______________________________________

For viewing changes to documentation in the repo, do the following:

.. code-block:: bash

    pip install -r docs/requirements.txt
    sphinx-reload docs/

This starts and opens a local version of the documentation in your browser (at localhost:5500/index.html) and keeps it up to date with any changes made. Note that changes to docstrings in the code will not trigger an update, only changes to the .rst files in the docs/ folder.

If you have added a feature or fixed a bug, add new tests to the appropriate file in the tests/ directory to test that the feature is working or that the bug is fixed. Before committing changes to the code, run all tests from the project root directory to ensure they are passing.

.. code-block:: bash

    pytest -n 4

Additionally, rerun the TutorialNotebook in Jupyter (Kernel -> Restart & Run All). Check that no unexpected behavior is encountered in these plots.

Now you are ready to commit your code. pre-commit should automatically run black, flake8, and isort to enforce style guidelines. If changes are made, the first commit will fail and you will need to stage the changes that have been made before committing again. If, for some reason, pre-commit fails to make changes to your files, you should be able to run the following to clean the files manually:

.. code-block:: bash

    black --skip-string-normalization --line-length=100 .
    flake8 .
    isort .

Adding new package dependencies to ldcpy
__________________________________

1) Adding new package dependencies requires updating the code in the following four places:

    /ci/environment.yml
    /ci/environment-dev.yml
    /ci/upstream-dev-environment.yml
    /requirements.txt

If the package dependency is specifically used for documentation, instead of adding it to /requirements.txt, add it to:

    /docs/source/requirements.txt

If this package is only used for documentation, skip the remaining steps.

2) If the package is one that includes C code (such as numpy or scipy), update the autodoc_mock_imports list in /docs/source/conf.py. The latest build of the documentation can be found at (https://readthedocs.org/projects/ldcpy/builds/), if the build fails and the error message indicates a problem with the newest package - try adding it to autodoc_mock_imports.

3) Finally, update the ldcpy-feedstock repository (git clone https://github.com/conda-forge/ldcpy-feedstock.git), or manually create a branch and add the dependency in the browser.
Name the branch add-<new_dependency_name>.
In the file /recipe/meta.yaml, in the "requirements" section, under "run", add your dependency to the list.

4) If the CI build encounters errors after adding a dependency, check the status of the CI workflow at (https://github.com/NCAR/ldcpy/actions?query=workflow%3ACI) to determine if the error is related to the new package.
