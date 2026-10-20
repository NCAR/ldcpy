#!/usr/bin/env python3
# flake8: noqa
"""Top-level module for ldcpy."""
from importlib.metadata import PackageNotFoundError, version

from .calcs import Datasetcalcs, Diffcalcs

# from .collect_datasets import collect_datasets
from .comp_checker import CompChecker
from .derived_vars import cam_budgets
from .plot import plot
from .util import (
    check_metrics,
    collect_datasets,
    compare_stats,
    open_datasets,
    save_metrics,
    subset_data,
    combine_datasets,
)

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = 'unknown'  # pragma: no cover
