import copy
import os
from typing import Optional

import cftime
import cv2
import matplotlib as mpl
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib import pyplot as plt
from numpy import inf
from scipy import stats as ss
from xrft import dft


class DatasetMetrics(object):
    """
    This class contains metrics for each point of a dataset after aggregating across one or more dimensions, and a method to access these metrics.
    """

    def __init__(
        self,
        ds: xr.DataArray,
        aggregate_dims: list,
    ):
        self._ds = ds if (ds.dtype == np.float64) else ds.astype(np.float64)
        # For some reason, casting to float64 removes all attrs from the dataset
        self._ds.attrs = ds.attrs

        # array metrics
        self._ns_con_var = None
        self._ew_con_var = None
        self._mean = None
        self._mean_abs = None
        self._std = None
        self._ddof = 1
        self._prob_positive = None
        self._odds_positive = None
        self._prob_negative = None
        self._zscore = None
        self._mae_day_max = None
        self._lag1 = None
        self._lag1_first_difference = None
        self._agg_dims = aggregate_dims
        self._quantile_value = None
        self._mean_squared = None
        self._root_mean_squared = None
        self._sum = None
        self._sum_squared = None
        self._variance = None
        self._pooled_variance = None
        self._pooled_variance_ratio = None
        self._standardized_mean = None
        self._quantile = 0.5
        self._spre_tol = 1.0e-4
        self._max_abs = None
        self._min_abs = None
        self._d_range = None
        self._min_val = None
        self._max_val = None
        self._grouping = None
        self._annual_harmonic_relative_ratio = None

        # single value metrics
        self._zscore_cutoff = None
        self._zscore_percent_significant = None

        self._frame_size = 1
        if aggregate_dims is not None:
            for dim in aggregate_dims:
                self._frame_size *= int(self._ds.sizes[dim])

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    def _con_var(self, dir, dataset) -> np.ndarray:
        if dir == 'ns':
            lat_length = dataset.sizes['lat']
            o_1, o_2 = xr.align(
                dataset.head({'lat': lat_length - 1}),
                dataset.tail({'lat': lat_length - 1}),
                join='override',
            )
        elif dir == 'ew':
            lon_length = dataset.sizes['lon']
            o_1, o_2 = xr.align(
                dataset,
                xr.concat(
                    [dataset.tail({'lon': lon_length - 1}), dataset.head({'lon': 1})],
                    dim='lon',
                ),
                join='override',
            )
        # con_var = xr.ufuncs.square((o_1 - o_2))
        con_var = np.square((o_1 - o_2))
        return con_var

    @property
    def pooled_variance(self) -> np.ndarray:
        """
        The overall variance of the dataset
        """
        if not self._is_memoized('_pooled_variance'):
            self._pooled_variance = self._ds.var(self._agg_dims).mean()
            self._pooled_variance.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._pooled_variance.attrs['units'] = f'{self._ds.units}$^2$'

        return self._pooled_variance

    @property
    def ns_con_var(self) -> np.ndarray:
        """
        The North-South Contrast Variance averaged along the aggregate dimensions
        """
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var = self._con_var('ns', self._ds).mean(self._agg_dims)
            self._ns_con_var.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._ns_con_var.attrs['units'] = f'{self._ds.units}$^2$'

        return self._ns_con_var

    @property
    def ew_con_var(self) -> np.ndarray:
        """
        The East-West Contrast Variance averaged along the aggregate dimensions
        """
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var = self._con_var('ew', self._ds).mean(self._agg_dims)
            self._ew_con_var.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._ew_con_var.attrs['units'] = f'{self._ds.units}$^2$'

        return self._ew_con_var

    @property
    def mean(self) -> np.ndarray:
        """
        The mean along the aggregate dimensions
        """
        if not self._is_memoized('_mean'):
            self._mean = self._ds.mean(self._agg_dims, skipna=True)
            self._mean.attrs = self._ds.attrs

        return self._mean

    @property
    def mean_abs(self) -> np.ndarray:
        """
        The mean of the absolute errors along the aggregate dimensions
        """
        if not self._is_memoized('_mean_abs'):
            self._mean_abs = abs(self._ds).mean(self._agg_dims, skipna=True)
            self._mean_abs.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._mean_abs.attrs['units'] = f'{self._ds.units}'

        return self._mean_abs

    @property
    def mean_squared(self) -> np.ndarray:
        """
        The absolute value of the mean along the aggregate dimensions
        """
        if not self._is_memoized('_mean_squared'):
            self._mean_squared = np.square(self.mean)
            self.mean_abs.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self.mean_abs.attrs['units'] = f'{self._ds.units}$^2$'

        return self._mean_squared

    @property
    def root_mean_squared(self) -> np.ndarray:
        """
        The absolute value of the mean along the aggregate dimensions
        """
        if not self._is_memoized('_root_mean_squared'):
            self._root_mean_squared = np.sqrt(np.square(self._ds).mean(dim=self._agg_dims))
            self._root_mean_squared.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._root_mean_squared.attrs['units'] = f'{self._ds.units}'

        return self._root_mean_squared

    @property
    def sum(self) -> np.ndarray:
        if not self._is_memoized('_sum'):
            self._sum = self._ds.sum(dim=self._agg_dims, skipna=True)
            self._sum.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._sum.attrs['units'] = f'{self._ds.units}'

        return self._sum

    @property
    def sum_squared(self) -> np.ndarray:
        if not self._is_memoized('_sum_squared'):
            self._sum_squared = np.square(self._sum_squared)
            self._sum_squared.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._sum_squared.attrs['units'] = f'{self._ds.units}$^2$'

        return self._sum_squared

    @property
    def std(self) -> np.ndarray:
        """
        The standard deviation along the aggregate dimensions
        """
        if not self._is_memoized('_std'):
            self._std = self._ds.std(self._agg_dims, ddof=self._ddof, skipna=True)
            self._std.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._std.attrs['units'] = ''

        return self._std

    @property
    def standardized_mean(self) -> np.ndarray:
        """
        The mean at each point along the aggregate dimensions divided by the standard deviation
        NOTE: will always be 0 if aggregating over all dimensions
        """
        if not self._is_memoized('_standardized_mean'):
            if self._grouping is None:
                self._standardized_mean = (self.mean - self._ds.mean()) / self._ds.std(ddof=1)
            else:
                self._standardized_mean = (
                    self.mean.groupby(self._grouping).mean()
                    - self.mean.groupby(self._grouping).mean().mean()
                ) / self.mean.groupby(self._grouping).mean().std(ddof=1)
            if hasattr(self._ds, 'units'):
                self._standardized_mean.attrs['units'] = ''

        return self._standardized_mean

    @property
    def variance(self) -> np.ndarray:
        """
        The variance along the aggregate dimensions
        """
        if not self._is_memoized('_variance'):
            self._variance = self._ds.var(self._agg_dims, skipna=True)
            self._variance.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._variance.attrs['units'] = f'{self._ds.units}$^2$'

        return self._variance

    @property
    def pooled_variance_ratio(self) -> np.ndarray:
        """
        The pooled variance along the aggregate dimensions
        """
        if not self._is_memoized('_pooled_variance_ratio'):
            self._pooled_variance_ratio = self.variance / self.pooled_variance
            self._pooled_variance_ratio.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._pooled_variance_ratio.attrs['units'] = ''

        return self._pooled_variance_ratio

    @property
    def prob_positive(self) -> np.ndarray:
        """
        The probability that a point is positive
        """
        if not self._is_memoized('_prob_positive'):
            self._prob_positive = (self._ds > 0).sum(self._agg_dims) / self._frame_size
            self._prob_positive.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._prob_positive.attrs['units'] = ''
        return self._prob_positive

    @property
    def prob_negative(self) -> np.ndarray:
        """
        The probability that a point is negative
        """
        if not self._is_memoized('_prob_negative'):
            self._prob_negative = (self._ds < 0).sum(self._agg_dims) / self._frame_size
            self._prob_negative.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._prob_negative.attrs['units'] = ''
        return self._prob_negative

    @property
    def odds_positive(self) -> np.ndarray:
        """
        The odds that a point is positive = prob_positive/(1-prob_positive)
        """
        if not self._is_memoized('_odds_positive'):
            if self._grouping is not None:
                self._odds_positive = self.prob_positive.groupby(self._grouping).mean() / (
                    1 - self.prob_positive.groupby(self._grouping).mean()
                )
            else:
                self._odds_positive = self.prob_positive / (1 - self.prob_positive)
            self._odds_positive.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._odds_positive.attrs['units'] = ''
        return self._odds_positive

    @property
    def zscore(self) -> np.ndarray:
        """
        The z-score of a point averaged along the aggregate dimensions under the null hypothesis that the true mean is zero.
        NOTE: currently assumes we are aggregating along the time dimension so is only suitable for a spatial plot.
        """
        if not self._is_memoized('_zscore'):
            self._zscore = np.divide(self.mean, self.std / np.sqrt(self._ds.sizes['time']))
            self._zscore.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._zscore.attrs['units'] = ''

        return self._zscore

    @property
    def mae_day_max(self) -> xr.DataArray:
        """
        The day of maximum mean absolute value at the point.
        NOTE: only available in spatial and spatial comparison plots
        """
        if not self._is_memoized('_mae_day_max'):
            self._mae_day_max = 0
            self._test = abs(self._ds).groupby('time.dayofyear').mean()
            self._mae_day_max = self._test.idxmax(dim='dayofyear')
            self._mae_day_max.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._mae_day_max.attrs['units'] = f'{self._ds.units}'

        return self._mae_day_max

    @property
    def quantile(self):
        return self._quantile

    @quantile.setter
    def quantile(self, q):
        self._quantile = q

    @property
    def spre_tol(self):
        return self._spre_tol

    @spre_tol.setter
    def spre_tol(self, t):
        self._spre_tol = t

    @property
    def quantile_value(self) -> xr.DataArray:
        self._quantile_value = self._ds.quantile(self.quantile, dim=self._agg_dims)
        self._quantile_value.attrs = self._ds.attrs
        if hasattr(self._ds, 'units'):
            self._quantile_value.attrs['units'] = ''

        return self._quantile_value

    @property
    def max_abs(self) -> xr.DataArray:
        if not self._is_memoized('_max_abs'):
            self._max_abs = abs(self._ds).max(dim=self._agg_dims)
            self._max_abs.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._max_abs.attrs['units'] = f'{self._ds.units}'

        return self._max_abs

    @property
    def min_abs(self) -> xr.DataArray:
        if not self._is_memoized('_min_abs'):
            self._min_abs = abs(self._ds).min(dim=self._agg_dims)
            self._min_abs.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._min_abs.attrs['units'] = f'{self._ds.units}'

        return self._min_abs

    @property
    def max_val(self) -> xr.DataArray:
        if not self._is_memoized('_max_val'):
            self._max_val = self._ds.max(dim=self._agg_dims)
            self._max_val.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._max_val.attrs['units'] = f'{self._ds.units}'

        return self._max_val

    @property
    def min_val(self) -> xr.DataArray:
        if not self._is_memoized('_min_val'):
            self._min_val = self._ds.min(dim=self._agg_dims)
            self._min_val.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._min_val.attrs['units'] = f'{self._ds.units}'

        return self._min_val

    @property
    def dyn_range(self) -> xr.DataArray:
        if not self._is_memoized('_range'):
            self._dyn_range = abs((self._ds).max() - (self._ds).min())
            self._dyn_range.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._dyn_range.attrs['units'] = f'{self._ds.units}'

        return self._dyn_range

    @property
    def lag1(self) -> xr.DataArray:
        """
        The deseasonalized lag-1 autocorrelation value by day of year
        NOTE: This metric returns an array of spatial values as the data set regardless of aggregate dimensions,
        so can only be plotted in a spatial plot.
        """
        if not self._is_memoized('_lag1'):
            self._deseas_resid = self._ds.groupby('time.dayofyear') - self._ds.groupby(
                'time.dayofyear'
            ).mean(dim='time')

            time_length = self._deseas_resid.sizes['time']
            current = self._deseas_resid.head({'time': time_length - 1})
            next = self._deseas_resid.shift({'time': -1}).head({'time': time_length - 1})

            num = current.dot(next, dims='time')
            denom = current.dot(current, dims='time')
            self._lag1 = num / denom

            self._lag1.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._lag1.attrs['units'] = ''

        return self._lag1

    @property
    def lag1_first_difference(self) -> xr.DataArray:
        """
        The deseasonalized lag-1 autocorrelation value of the first difference of the data by day of year
        NOTE: This metric returns an array of spatial values as the data set regardless of aggregate dimensions,
        so can only be plotted in a spatial plot.
        """
        if not self._is_memoized('_lag1_first_difference'):
            self._deseas_resid = self._ds.groupby('time.dayofyear') - self._ds.groupby(
                'time.dayofyear'
            ).mean(dim='time')
            # self._deseas_resid=self._ds

            time_length = self._deseas_resid.sizes['time']
            current = self._deseas_resid.head({'time': time_length - 1})
            next = self._deseas_resid.shift({'time': -1}).head({'time': time_length - 1})
            first_difference = next - current
            first_difference_current = first_difference.head({'time': time_length - 1})
            first_difference_next = first_difference.shift({'time': -1}).head(
                {'time': time_length - 1}
            )

            # num = first_difference_current.dot(first_difference_next, dims='time')
            num = (first_difference_current * first_difference_next).sum(dim=['time'], skipna=True)
            denom = first_difference_current.dot(first_difference_current, dims='time')
            self._lag1_first_difference = num / denom

            self._lag1_first_difference.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._lag1_first_difference.attrs['units'] = ''

        return self._lag1_first_difference

    @property
    def annual_harmonic_relative_ratio(self) -> xr.DataArray:
        """
        The annual harmonic relative to the average periodogram value
        in a neighborhood of 50 frequencies around the annual frequency
        NOTE: This assumes the values along the "time" dimension are equally spaced.
        NOTE: This metric returns a lat-lon array regardless of aggregate dimensions, so can only be used in a spatial plot.
        """
        if not self._is_memoized('_annual_harmonic_relative_ratio'):
            # drop time coordinate labels or else it will try to parse them as numbers to check spacing and fail
            ds_copy = self._ds
            new_index = [i for i in range(0, self._ds.time.size)]
            new_ds = ds_copy.assign_coords({'time': new_index})

            DF = dft(new_ds, dim=['time'], detrend='constant')
            S = np.real(DF * np.conj(DF) / self._ds.sizes['time'])
            S_annual = S.isel(
                freq_time=int(self._ds.sizes['time'] / 2) + int(self._ds.sizes['time'] / 365)
            )  # annual power
            neighborhood = (
                int(self._ds.sizes['time'] / 2) + int(self._ds.sizes['time'] / 365) - 25,
                int(self._ds.sizes['time'] / 2) + int(self._ds.sizes['time'] / 365) + 25,
            )
            S_mean = xr.concat(
                [
                    S.isel(
                        freq_time=slice(
                            max(0, neighborhood[0]),
                            int(self._ds.sizes['time'] / 2) + int(self._ds.sizes['time'] / 365) - 1,
                        )
                    ),
                    S.isel(
                        freq_time=slice(
                            int(self._ds.sizes['time'] / 2) + int(self._ds.sizes['time'] / 365) + 1,
                            neighborhood[1],
                        )
                    ),
                ],
                dim='freq_time',
            ).mean(dim='freq_time')
            ratio = S_annual / S_mean
            self._annual_harmonic_relative_ratio = ratio

            if hasattr(self._ds, 'units'):
                self._annual_harmonic_relative_ratio.attrs['units'] = ''
        return self._annual_harmonic_relative_ratio

    @property
    def zscore_cutoff(self) -> np.ndarray:
        """
        The Z-Score cutoff for a point to be considered significant
        """
        if not self._is_memoized('_zscore_cutoff'):
            pvals = 2 * (1 - ss.norm.cdf(np.abs(self.zscore)))
            if isinstance(pvals, np.float64):
                pvals_array = np.array(pvals)
                sorted_pvals = pvals_array
            else:
                pvals_array = pvals
                sorted_pvals = np.sort(pvals_array).flatten()
            fdr_zscore = 0.01
            p = np.argwhere(
                sorted_pvals <= fdr_zscore * np.arange(1, pvals_array.size + 1) / pvals_array.size
            )
            pval_cutoff = np.empty(0)
            if not len(p) == 0:
                pval_cutoff = sorted_pvals[p[len(p) - 1]]
            if not (pval_cutoff.size == 0):
                zscore_cutoff = ss.norm.ppf(1 - pval_cutoff)
            else:
                zscore_cutoff = 'na'
            self._zscore_cutoff = zscore_cutoff

            return self._zscore_cutoff

    @property
    def annual_harmonic_relative_ratio_pct_sig(self) -> np.ndarray:
        """
        The percentage of points past the significance cutoff (p value <= 0.01) for the
        annual harmonic relative to the average periodogram value
        in a neighborhood of 50 frequencies around the annual frequency
        """

        pvals = 1 - ss.f.cdf(self.annual_harmonic_relative_ratio, 2, 100)
        sorted_pvals = np.sort(pvals)
        if len(sorted_pvals[sorted_pvals <= 0.01]) == 0:
            return 0
        sig_cutoff = ss.f.ppf(1 - max(sorted_pvals[sorted_pvals <= 0.01]), 2, 50)
        pct_sig = 100 * np.mean(self.annual_harmonic_relative_ratio > sig_cutoff)
        return pct_sig

    @property
    def zscore_percent_significant(self) -> np.ndarray:
        """
        The percent of points where the zscore is considered significant
        """
        if not self._is_memoized('_zscore_percent_significant'):
            pvals = 2 * (1 - ss.norm.cdf(np.abs(self.zscore)))
            if isinstance(pvals, np.float64):
                pvals_array = np.array(pvals)
                sorted_pvals = pvals_array
            else:
                pvals_array = pvals
                sorted_pvals = np.sort(pvals_array).flatten()
            fdr_zscore = 0.01
            p = np.argwhere(sorted_pvals <= fdr_zscore * np.arange(1, pvals.size + 1) / pvals.size)
            pval_cutoff = sorted_pvals[p[len(p) - 1]]
            if not (pval_cutoff.size == 0):
                sig_locs = np.argwhere(pvals <= pval_cutoff)
                percent_sig = 100 * np.size(sig_locs, 0) / pvals.size
            else:
                percent_sig = 0
            self._zscore_percent_significant = percent_sig

            return self._zscore_percent_significant

    def get_metric(self, name: str, q: Optional[int] = 0.5, grouping: Optional[str] = None, ddof=1):
        """
        Gets a metric aggregated across one or more dimensions of the dataset

        Parameters
        ==========
        name : str
            The name of the metric (must be identical to a property name)

        q: float, optional
           (default 0.5)

        Returns
        =======
        out : xarray.DataArray
            A DataArray of the same size and dimensions the original dataarray,
            minus those dimensions that were aggregated across.
        """
        if isinstance(name, str):
            if name == 'ns_con_var':
                return self.ns_con_var
            if name == 'ew_con_var':
                return self.ew_con_var
            if name == 'mean':
                return self.mean
            if name == 'std':
                self._ddof = ddof
                return self.std
            if name == 'standardized_mean':
                self._grouping = grouping
                return self.standardized_mean
            if name == 'variance':
                return self.variance
            if name == 'pooled_var_ratio':
                return self.pooled_variance_ratio
            if name == 'prob_positive':
                return self.prob_positive
            if name == 'prob_negative':
                return self.prob_negative
            if name == 'odds_positive':
                self._grouping = grouping
                return self.odds_positive
            if name == 'zscore':
                return self.zscore
            if name == 'mae_day_max':
                return self.mae_day_max
            if name == 'mean_abs':
                return self.mean_abs
            if name == 'mean_squared':
                return self.mean_squared
            if name == 'rms':
                return self.root_mean_squared
            if name == 'sum':
                return self.sum
            if name == 'sum_squared':
                return self.sum_squared
            if name == 'ann_harmonic_ratio':
                return self.annual_harmonic_relative_ratio
            if name == 'quantile':
                self.quantile = q
                return self.quantile_value
            if name == 'lag1':
                return self.lag1
            if name == 'lag1_first_difference':
                return self.lag1_first_difference
            if name == 'max_abs':
                return self.max_abs
            if name == 'min_abs':
                return self.min_abs
            if name == 'max_val':
                return self.max_val
            if name == 'min_val':
                return self.min_val
            if name == 'ds':
                return self._ds
            raise ValueError(f'there is no metric with the name: {name}.')
        else:
            raise TypeError('name must be a string.')

    def get_single_metric(self, name: str):
        """
        Gets a metric consisting of a single float value

        Parameters
        ==========
        name : str
            the name of the metric (must be identical to a property name)

        Returns
        =======
        out : float
            The metric value
        """
        if isinstance(name, str):
            if name == 'zscore_cutoff':
                return self.zscore_cutoff
            if name == 'zscore_percent_significant':
                return self.zscore_percent_significant
            if name == 'range':
                return self.dyn_range
            if name == 'spre_tol':
                return self.spre_tol
            if name == 'pooled_variance':
                return self.pooled_variance
            if name == 'annual_harmonic_relative_ratio_pct_sig':
                return self.annual_harmonic_relative_ratio_pct_sig
            raise ValueError(f'there is no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class DiffMetrics(object):
    """
    This class contains metrics on the overall dataset that require more than one input dataset to compute
    """

    def __init__(
        self,
        ds1: xr.DataArray,
        ds2: xr.DataArray,
        aggregate_dims: Optional[list] = None,
    ) -> None:
        if isinstance(ds1, xr.DataArray):
            # Datasets
            self._ds1 = ds1

        if isinstance(ds2, xr.DataArray):
            # Datasets
            self._ds2 = ds2

        else:
            raise TypeError(
                f'ds must be of type xarray.DataArray. Type(s): {str(type(ds1))} {str(type(ds2))}'
            )

        self._metrics1 = DatasetMetrics(self._ds1, aggregate_dims)
        self._metrics2 = DatasetMetrics(self._ds2, aggregate_dims)
        self._aggregate_dims = aggregate_dims
        self._pcc = None
        self._covariance = None
        self._ks_p_value = None
        self._n_rms = None
        self._n_emax = None
        self._rel_spatial_error = None
        self._ssim_value = None

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    @property
    def covariance(self) -> np.ndarray:
        """
        The covariance between the two datasets
        """
        if not self._is_memoized('_covariance'):
            self._covariance = (
                (self._metrics2.get_metric('ds') - self._metrics2.get_metric('mean'))
                * (self._metrics1.get_metric('ds') - self._metrics1.get_metric('mean'))
            ).mean()

        return self._covariance

    @property
    def ks_p_value(self):
        """
        The Kolmogorov-Smirnov p-value
        """
        # Note: ravel() foces a compute for dask, but ks test in scipy can't
        # work with uncomputed dask arrays
        if not self._is_memoized('_ks_p_value'):
            self._ks_p_value = np.asanyarray(ss.ks_2samp(np.ravel(self._ds2), np.ravel(self._ds1)))
        return self._ks_p_value[1]

    @property
    def pearson_correlation_coefficient(self):
        """
        returns the pearson correlation coefficient between the two datasets
        """
        if not self._is_memoized('_pearson_correlation_coefficient'):
            self._pcc = (
                self.covariance
                / self._metrics1.get_metric('std', ddof=0)
                / self._metrics2.get_metric('std', ddof=0)
            )

        return self._pcc

    @property
    def normalized_max_pointwise_error(self):
        """
        The absolute value of the maximum pointwise difference, normalized
        by the range of values for the first set
        """
        if not self._is_memoized('_normalized_max_pointwise_error'):
            tt = abs((self._metrics1.get_metric('ds') - self._metrics2.get_metric('ds')).max())
            self._n_emax = tt / self._metrics1.dyn_range

        return self._n_emax

    @property
    def normalized_root_mean_squared(self):
        """
        The absolute value of the mean along the aggregate dimensions, normalized
        by the range of values for the first set
        """
        if not self._is_memoized('_normalized_root_mean_squared'):
            tt = np.sqrt(
                np.square(self._metrics1.get_metric('ds') - self._metrics2.get_metric('ds')).mean(
                    dim=self._aggregate_dims
                )
            )
            self._n_rms = tt / self._metrics1.dyn_range

        return self._n_rms

    @property
    def spatial_rel_error(self):
        """
        At each grid point, we compute the relative error.  Then we report the percentage of grid point whose
        relative error is above the specified tolerance (1e-4 by default).
        """

        if not self._is_memoized('_spatial_rel_error'):
            # print(self._metrics1.get_metric('ds').shape)
            sp_tol = self._metrics1.spre_tol
            # unraveling converts the dask array to numpy, but then
            # we can assign the 1.0 and avoid zero (couldn't figure another way)
            t1 = np.ravel(self._metrics1.get_metric('ds'))
            t2 = np.ravel(self._metrics2.get_metric('ds'))
            # check for zeros in t1 (if zero then change to 1 - which
            # does an absolute error at that point)
            z = np.where(abs(t1) == 0)
            t1[z] = 1.0
            # we don't want to use nan (ocassionally in cam data - often oin ocn)
            m_t2 = np.ma.masked_invalid(t2).compressed()
            m_t1 = np.ma.masked_invalid(t1).compressed()
            m_tt = m_t1 - m_t2
            m_tt = m_tt / m_t1
            a = len(m_tt[m_tt > sp_tol])
            sz = m_tt.shape[0]

            self._spatial_rel_error = (a / sz) * 100

        return self._spatial_rel_error

    @property
    def ssim_value(self):
        """
        We compute the SSIM (structural similarity index) on the visualization of the spatial data.
        """
        if not self._is_memoized('_ssim'):
            d1 = self._metrics1.get_metric('ds')
            d2 = self._metrics2.get_metric('ds')
            lat1 = d1['lat']
            lat2 = d2['lat']
            cy1, lon1 = add_cyclic_point(d1, coord=d1['lon'])
            cy2, lon2 = add_cyclic_point(d2, coord=d2['lon'])

            # Prevent showing stuff
            backend_ = mpl.get_backend()
            mpl.use('Agg')

            no_inf_d1 = np.nan_to_num(cy1, nan=np.nan)
            no_inf_d2 = np.nan_to_num(cy2, nan=np.nan)

            color_min = min(
                np.min(d1.where(d1 != -inf)).values.min(),
                np.min(d2.where(d2 != -inf)).values.min(),
            )
            color_max = max(
                np.max(d1.where(d1 != inf)).values.max(),
                np.max(d2.where(d2 != inf)).values.max(),
            )

            fig = plt.figure(dpi=300, figsize=(9, 2.5))

            mymap = copy.copy(plt.cm.get_cmap('coolwarm'))
            mymap.set_under(color='black')
            mymap.set_over(color='white')
            mymap.set_bad(alpha=0)

            ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=0.0))
            ax1.set_facecolor('#39ff14')

            ax1.pcolormesh(
                lon1,
                lat1,
                no_inf_d1,
                transform=ccrs.PlateCarree(),
                cmap=mymap,
                vmin=color_min,
                vmax=color_max,
            )
            ax1.set_global()
            ax1.coastlines(linewidth=0.5)
            ax1.axis('off')
            plt.margins(0, 0)
            extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            ax1.imshow
            plt.savefig('t_ssim1', bbox_inches=extent1, transparent=True, pad_inches=0)
            ax1.axis('on')

            ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=0.0))
            ax2.set_facecolor('#39ff14')

            ax2.pcolormesh(
                lon2,
                lat2,
                no_inf_d2,
                transform=ccrs.PlateCarree(),
                cmap=mymap,
                vmin=color_min,
                vmax=color_max,
            )
            ax2.set_global()
            ax2.coastlines(linewidth=0.5)
            plt.margins(0, 0)
            ax2.imshow
            ax2.axis('off')
            extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            plt.savefig('t_ssim2', bbox_inches=extent2, transparent=True, pad_inches=0)

            ax2.axis('on')

            from skimage.metrics import structural_similarity as ssim

            img1 = cv2.imread('t_ssim1.png')
            img2 = cv2.imread('t_ssim2.png')
            # print(img1.shape)
            # print(img2.shape)
            s = ssim(img1, img2, multichannel=True)
            if os.path.exists('t_ssim1.png'):
                os.remove('t_ssim1.png')
            if os.path.exists('t_ssim2.png'):
                os.remove('t_ssim2.png')

            # Reset backend
            mpl.use(backend_)

            self._ssim_value = s

        return self._ssim_value

    def get_diff_metric(self, name: str):
        """
        Gets a metric on the dataset that requires more than one input dataset

        Parameters
        ==========
        name : str
            The name of the metric (must be identical to a property name)

        Returns
        =======
        out : float
        """
        if isinstance(name, str):
            if name == 'pearson_correlation_coefficient':
                return self.pearson_correlation_coefficient
            if name == 'covariance':
                return self.covariance
            if name == 'ks_p_value':
                return self.ks_p_value
            if name == 'n_rms':
                return self.normalized_root_mean_squared
            if name == 'n_emax':
                return self.normalized_max_pointwise_error
            if name == 'spatial_rel_error':
                return self.spatial_rel_error
            if name == 'ssim':
                return self.ssim_value
            raise ValueError(f'there is no metric with the name: {name}.')
        else:
            raise TypeError('name must be a string.')
