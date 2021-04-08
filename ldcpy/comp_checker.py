# This is intended to be used to check a particular
# API for checking metric during compression


# Notes:
# - if initial try passes, move on or try more aggressive?


from math import exp, pi, sqrt

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import HTML, display

from ldcpy import calcs as lm


class CompChecker:
    """
    This class contains code to check a quantity against a tolerance
    for a time slice and recommend whether to try a different
    compression level.

    When you initialize the object, the following paramters are optional:

    calc_type: str, optional
          A valid Diffcalcs options ("ssim_fp" is the default)
    cal_tol: float
          The tolerance to check for that calc_type (default is .9995)
    tol_greater_than: boolean, optional
          If true, to pass, the calculated value should be greater than
          the calc_tol (default is True)
    compressor: str, optional
          Name of the compressor (options: "zfp") (default is "zfp")
    comp_mode: str, optional
        Corresponds to the mode of the compressor (default is "p")
    accept_first: boolean, optional
        If true, then if the first comp level tried passes, accept that
        as optimal (do not see if a more aggressive level could be used)
        (default is False)
    """

    def __init__(
        self,
        calc_type='ssim_fp',
        calc_tol=0.9995,
        tol_greater_than=True,
        compressor='zfp',
        comp_mode='p',
        accept_first=False,
    ):

        self._calc_type = calc_type
        self._calc_tol = calc_tol
        self._tol_greater_than = True
        self._compressor = compressor
        self._comp_mode = comp_mode
        self._accept_first = accept_first

        self._counter = 0
        self._results_dict = {}
        self._prev_level = -1
        self._prev_pass = False
        self._opt_level = None

    def reset_checker(self):  # call before doing the next timestep
        self._counter = 0
        self._results_dict = {}
        self._prev_level = -1
        self._prev_pass = False
        self._opt_level = None

    def eval_comp_level(self, orig_da, comp_da, comp_level):

        dc = lm.Diffcalcs(orig_da, comp_da)
        val = dc.get_diff_calc(self._calc_type)

        if self._tol_greater_than:
            if val >= self._calc_tol:
                level_passed = True
            else:
                level_passed = False
        else:
            if val <= self._calc_tol:
                level_passed = True
            else:
                level_passed = False

        opt_level = None
        call_again = True
        # keep track for now (for debugging)
        self._results_dict[comp_level] = [val, level_passed]

        if self._counter == 0:  # first try
            if level_passed:  # this one passed
                self._prev_level = comp_level
                self._prev_pass = True
                if self._accept_first:  # ok to accept first value (done!)
                    # (instead of auto-checking if we can go more aggressive)
                    opt_level = comp_level
                    call_again = False
            else:  # this one failed
                self._prev_level = comp_level
                self._prev_pass = False

            if call_again:
                new_level = self._comp_rules(comp_level, level_passed)

        else:  # counter > 0
            if level_passed:  # this one passed
                if self._prev_pass:
                    self._prev_level = comp_level
                    new_level = self._comp_rules(comp_level, level_passed)
                    if new_level == comp_level:  # no change - reached min (done!)
                        opt_level = new_level
                        call_again = False
                else:  # prev fail and this one passed, so done!
                    opt_level = comp_level
                    call_again = False
            else:  # this one failed
                if self._prev_pass:  # prev passed and this failed, so done!
                    opt_level = self._prev_level
                    call_again = False
                else:  # last failed also
                    self._prev_level = comp_level
                    new_level = self._comp_rules(comp_level, level_passed)
                    if new_level == comp_level:  # no change- reached max (done!)
                        opt_level = -1
                        call_again = False

        # increment counter
        self._counter = self._counter + 1
        if opt_level:
            self._opt_level = opt_level
        if call_again:
            self._new_level = new_level

        # RETURN (if call_again == False, then opt_level is set).
        # if opt_level = -1 then use lossless
        # if calling again, then a new level is suggested
        return call_again

    def get_opt_level(self):
        return self._opt_level

    def get_new_level(self):
        return self._new_level

    def show_results(self):
        my_cols = [self._calc_type, 'Passed?']

        df = pd.DataFrame.from_dict(self._results_dict, orient='index', columns=my_cols)
        a = self._compressor
        display(HTML('<br>'))
        display(HTML(f'<span style="color:green">{a} level results: </span>  '))
        display(df)

    def _comp_rules(self, comp_level, level_passed):
        if self._compressor == 'zfp':
            new_level = self._zfp_rules(comp_level, level_passed)
        else:
            print('Rules not defined for compressor = ', self._compressor)
            new_level = -1
        return new_level

    def _zfp_rules(self, comp_level, level_passed):

        if self._comp_mode == 'p':  # precision
            pmax = 28
            pmin = 6
            if level_passed:  # passed, so go more aggressive
                new_level = comp_level - 2
                new_level = max(new_level, pmin)
            else:  # failed, so more conservative
                new_level = comp_level + 2
                new_level = min(new_level, pmax)
        else:
            print('No rules for zfp mode = ', self._comp_mode)
            new_level = -1
        return new_level
