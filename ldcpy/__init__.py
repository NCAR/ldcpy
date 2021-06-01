#!/usr/bin/env python3
# flake8: noqa
""" Top-level module for ldcpy. """
from pkg_resources import DistributionNotFound, get_distribution

from .calcs import Datasetcalcs, Diffcalcs
from .comp_checker import CompChecker
from .convert import CalendarDateTime
from .derived_vars import cam_budgets
from .plot import plot
from .util import check_metrics, collect_datasets, combine_datasets, compare_stats, open_datasets

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:  # pragma: no cover
    __version__ = 'unknown'  # pragma: no cover
