import copy
from math import exp, pi, sqrt
from typing import Optional

import cf_xarray as cf
import dask
import matplotlib as mpl
import numpy as np
import xarray as xr
from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib import pyplot as plt
from scipy import stats as ss
from xrft import dft


class DatasetMetrics:
    """
    This class contains metrics for each point of a dataset after aggregating across one or more dimensions, and a method to access these metrics. Expects a DataArray.

    """

    def __init__(
        self,
        ds: xr.DataArray,
        aggregate_dims: list,
        time_dim_name: str = 'time',
        lat_dim_name: str = None,
        lon_dim_name: str = None,
        vert_dim_name: str = None,
        lat_coord_name: str = None,
        lon_coord_name: str = None,
        q: float = 0.5,
        spre_tol: float = 1.0e-4,
    ):
        self._ds = ds if (ds.dtype == np.float64) else ds.astype(np.float64)
        # For some reason, casting to float64 removes all attrs from the dataset
        self._ds.attrs = ds.attrs

        # Let's just get all the lat/lon and time
        # names from the file if they are None
        # lon coord
        if lon_coord_name is None:
            lon_coord_name = ds.cf.coordinates['longitude'][0]
        self._lon_coord_name = lon_coord_name

        # lat coord
        if lat_coord_name is None:
            lat_coord_name = ds.cf.coordinates['latitude'][0]
        self._lat_coord_name = lat_coord_name

        dd = ds.cf['latitude'].dims
        ll = len(dd)
        if ll == 1:
            if lat_dim_name is None:
                lat_dim_name = dd[0]
            if lon_dim_name is None:
                lon_dim_name = ds.cf['longitude'].dims[0]
        elif ll == 2:
            if lat_dim_name is None:
                lat_dim_name = dd[0]
            if lon_dim_name is None:
                lon_dim_name = dd[1]

        self._latlon_dims = ll
        self._lat_dim_name = lat_dim_name
        self._lon_dim_name = lon_dim_name

        # vertical dimension?
        if vert_dim_name is None:
            vert = 'vertical' in ds.cf
            if vert:
                vert_dim_name = ds.cf['vertical'].name
        self._vert_dim_name = vert_dim_name

        # time dimension TO DO: check this (after cf_xarray update)
        self._time_dim_name = time_dim_name

        self._quantile = q
        self._spre_tol = spre_tol
        self._agg_dims = aggregate_dims
        self._frame_size = 1

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
        self._quantile_value = None
        self._mean_squared = None
        self._root_mean_squared = None
        self._sum = None
        self._sum_squared = None
        self._variance = None
        self._pooled_variance = None
        self._pooled_variance_ratio = None
        self._standardized_mean = None
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

        if aggregate_dims is not None:
            for dim in aggregate_dims:
                self._frame_size *= int(self._ds.sizes[dim])

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    def _con_var(self, dir, dataset) -> xr.DataArray:

        if dir == 'ns':
            tt = dataset.diff(self._lat_dim_name, 1)

        elif dir == 'ew':
            ds_h = xr.concat(
                [
                    dataset,
                    dataset.head({self._lon_dim_name: 1}),
                ],
                dim=self._lon_dim_name,
            )
            tt = ds_h.diff(self._lon_dim_name, 1)

        con_var = np.square(tt)
        return con_var

    @property
    def pooled_variance(self) -> xr.DataArray:
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
    def ns_con_var(self) -> xr.DataArray:
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
    def ew_con_var(self) -> xr.DataArray:
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
    def mean(self) -> xr.DataArray:
        """
        The mean along the aggregate dimensions
        """
        # print("mean")
        if not self._is_memoized('_mean'):
            self._mean = self._ds.mean(self._agg_dims, skipna=True)
            self._mean.attrs = self._ds.attrs
        return self._mean

    @property
    def mean_abs(self) -> xr.DataArray:
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
    def mean_squared(self) -> xr.DataArray:
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
    def root_mean_squared(self) -> xr.DataArray:
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
    def sum(self) -> xr.DataArray:
        if not self._is_memoized('_sum'):
            self._sum = self._ds.sum(dim=self._agg_dims, skipna=True)
            self._sum.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._sum.attrs['units'] = f'{self._ds.units}'

        return self._sum

    @property
    def sum_squared(self) -> xr.DataArray:
        if not self._is_memoized('_sum_squared'):
            self._sum_squared = np.square(self._sum_squared)
            self._sum_squared.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._sum_squared.attrs['units'] = f'{self._ds.units}$^2$'

        return self._sum_squared

    @property
    def std(self) -> xr.DataArray:
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
    def standardized_mean(self) -> xr.DataArray:
        """
        The mean at each point along the aggregate dimensions divided by the standard deviation
        NOTE: will always be 0 if aggregating over all dimensions
        """
        if not self._is_memoized('_standardized_mean'):
            if self._grouping is None:
                self._standardized_mean = (self.mean - self._ds.mean()) / self._ds.std(ddof=1)
            else:
                grouped = self.mean.groupby(self._grouping)
                grouped_mean = grouped.mean()
                self._standardized_mean = (grouped_mean - grouped_mean.mean()) / grouped_mean.std(
                    ddof=1
                )
            if hasattr(self._ds, 'units'):
                self._standardized_mean.attrs['units'] = ''

        return self._standardized_mean

    @property
    def variance(self) -> xr.DataArray:
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
    def pooled_variance_ratio(self) -> xr.DataArray:
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
    def prob_positive(self) -> xr.DataArray:
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
    def prob_negative(self) -> xr.DataArray:
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
    def odds_positive(self) -> xr.DataArray:
        """
        The odds that a point is positive = prob_positive/(1-prob_positive)
        """
        if not self._is_memoized('_odds_positive'):
            if self._grouping is not None:
                grouped = self.prob_positive.groupby(self._grouping)
                grouped_mean = grouped.mean()
                self._odds_positive = grouped_mean / (1 - grouped_mean)
            else:
                self._odds_positive = self.prob_positive / (1 - self.prob_positive)
            self._odds_positive.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._odds_positive.attrs['units'] = ''
        return self._odds_positive

    @property
    def zscore(self) -> xr.DataArray:
        """
        The z-score of a point averaged along the aggregate dimensions under the null hypothesis that the true mean is zero.
        NOTE: currently assumes we are aggregating along the time dimension so is only suitable for a spatial plot.
        """
        if not self._is_memoized('_zscore'):
            self._zscore = np.divide(
                self.mean, self.std / np.sqrt(self._ds.sizes[self._time_dim_name])
            )
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
            key = f'{self._time_dim_name}.dayofyear'
            self._mae_day_max = 0
            self._test = abs(self._ds).groupby(key).mean()
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
            key = f'{self._time_dim_name}.dayofyear'
            grouped = self._ds.groupby(key)
            self._deseas_resid = grouped - grouped.mean(dim=self._time_dim_name)
            time_length = self._deseas_resid.sizes[self._time_dim_name]
            current = self._deseas_resid.head({self._time_dim_name: time_length - 1})
            next = self._deseas_resid.shift({self._time_dim_name: -1}).head(
                {self._time_dim_name: time_length - 1}
            )

            num = current.dot(next, dims=self._time_dim_name)
            denom = current.dot(current, dims=self._time_dim_name)
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
            key = f'{self._time_dim_name}.dayofyear'
            grouped = self._ds.groupby(key)
            self._deseas_resid = grouped - grouped.mean(dim=self._time_dim_name)
            time_length = self._deseas_resid.sizes[self._time_dim_name]
            current = self._deseas_resid.head({self._time_dim_name: time_length - 1})
            next = self._deseas_resid.shift({self._time_dim_name: -1}).head(
                {self._time_dim_name: time_length - 1}
            )
            first_difference = next - current
            first_difference_current = first_difference.head({self._time_dim_name: time_length - 1})
            first_difference_next = first_difference.shift({self._time_dim_name: -1}).head(
                {self._time_dim_name: time_length - 1}
            )

            num = (first_difference_current * first_difference_next).sum(
                dim=[self._time_dim_name], skipna=True
            )
            denom = first_difference_current.dot(first_difference_current, dims=self._time_dim_name)
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

            new_index = [i for i in range(0, self._ds[self._time_dim_name].size)]
            new_ds = ds_copy.assign_coords({self._time_dim_name: new_index})

            lon_coord_name = self._lon_coord_name
            lat_coord_name = self._lat_coord_name

            DF = dft(new_ds, dim=[self._time_dim_name], detrend='constant')
            # the above does not preserve the lat/lon attributes
            DF[lon_coord_name].attrs = new_ds[lon_coord_name].attrs
            DF[lat_coord_name].attrs = new_ds[lat_coord_name].attrs
            DF.attrs = new_ds.attrs

            S = np.real(DF * np.conj(DF) / self._ds.sizes[self._time_dim_name])
            S_annual = S.isel(
                freq_time=int(self._ds.sizes[self._time_dim_name] / 2)
                + int(self._ds.sizes[self._time_dim_name] / 365)
            )  # annual power
            neighborhood = (
                int(self._ds.sizes[self._time_dim_name] / 2)
                + int(self._ds.sizes[self._time_dim_name] / 365)
                - 25,
                int(self._ds.sizes[self._time_dim_name] / 2)
                + int(self._ds.sizes[self._time_dim_name] / 365)
                + 25,
            )
            S_mean = xr.concat(
                [
                    S.isel(
                        freq_time=slice(
                            max(0, neighborhood[0]),
                            int(self._ds.sizes[self._time_dim_name] / 2)
                            + int(self._ds.sizes[self._time_dim_name] / 365)
                            - 1,
                        )
                    ),
                    S.isel(
                        freq_time=slice(
                            int(self._ds.sizes[self._time_dim_name] / 2)
                            + int(self._ds.sizes[self._time_dim_name] / 365)
                            + 1,
                            neighborhood[1],
                        )
                    ),
                ],
                dim='freq_time',
            ).mean(dim='freq_time')
            ratio = S_annual / S_mean

            # ratio.cf.describe()

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


class DiffMetrics:
    """
    This class contains metrics on the overall dataset that require more than one input dataset to compute
    """

    def __init__(
        self,
        ds1: xr.DataArray,
        ds2: xr.DataArray,
        aggregate_dims: Optional[list] = None,
        **metrics_kwargs,
    ) -> None:
        if isinstance(ds1, xr.DataArray) and isinstance(ds2, xr.DataArray):
            # Datasets
            self._ds1 = ds1
            self._ds2 = ds2
        else:
            raise TypeError(
                f'ds must be of type xarray.DataArray. Type(s): {str(type(ds1))} {str(type(ds2))}'
            )

        self._metrics1 = DatasetMetrics(self._ds1, aggregate_dims, **metrics_kwargs)
        self._metrics2 = DatasetMetrics(self._ds2, aggregate_dims, **metrics_kwargs)
        self._aggregate_dims = aggregate_dims
        self._pcc = None
        self._covariance = None
        self._ks_p_value = None
        self._n_rms = None
        self._n_emax = None
        self._spatial_rel_error = None
        self._ssim_value = None
        self._ssim_value_fp = None
        self._ssim_value_fp_old = None
        self._max_spatial_rel_error = None

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    # n is the box width (1D)
    # sigma is the radius for the gaussian
    def _oned_gauss(self, n, sigma):
        r = range(-int(n / 2), int(n / 2) + 1)
        return [(1 / (sigma * sqrt(2 * pi))) * exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]

    @property
    def covariance(self) -> xr.DataArray:
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
        # Note: ravel() forces a compute for dask, but ks test in scipy can't
        # work with uncomputed dask arrays
        if not self._is_memoized('_ks_p_value'):
            d1_p = (np.ravel(self._ds1)).astype('float64')
            d2_p = (np.ravel(self._ds2)).astype('float64')
            self._ks_p_value = np.asanyarray(ss.ks_2samp(d2_p, d1_p))
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
            sp_tol = self._metrics1.spre_tol
            # unraveling converts the dask array to numpy, but then
            # we can assign the 1.0 and avoid zero (couldn't figure another way)
            t1 = np.ravel(self._metrics1.get_metric('ds'))
            t2 = np.ravel(self._metrics2.get_metric('ds'))

            # check for zeros in t1 (if zero then change to 1 - which
            # does an absolute error at that point)
            z = (np.where(abs(t1) == 0))[0]
            if z.size > 0:
                t1_denom = np.copy(t1)
                t1_denom[z] = 1.0
            else:
                t1_denom = t1

            # we don't want to use nan
            # (ocassionally in cam data - often in ocn)
            m_t2 = np.ma.masked_invalid(t2).compressed()
            m_t1 = np.ma.masked_invalid(t1).compressed()

            if z.size > 0:
                m_t1_denom = np.ma.masked_invalid(t1_denom).compressed()
            else:
                m_t1_denom = m_t1

            m_tt = m_t1 - m_t2
            m_tt = m_tt / m_t1_denom

            # find the max spatial error also if None
            if self._max_spatial_rel_error is None:
                max_spre = np.max(m_tt)
                self._max_spatial_rel_error = max_spre

            # percentage greater than the tolerance
            a = len(m_tt[abs(m_tt) > sp_tol])
            sz = m_tt.shape[0]

            self._spatial_rel_error = (a / sz) * 100

        return self._spatial_rel_error

    @property
    def max_spatial_rel_error(self):
        """
        We compute the relative error at each grid point and return the maximun.
        """
        # this is also computed as part of the spatial rel error
        if not self._is_memoized('_max_spatial_rel_error'):
            t1 = np.ravel(self._metrics1.get_metric('ds'))
            t2 = np.ravel(self._metrics2.get_metric('ds'))
            # check for zeros in t1 (if zero then change to 1 - which
            # does an absolute error at that point)
            z = (np.where(abs(t1) == 0))[0]
            if z.size > 0:
                t1_denom = np.copy(t1)
                t1_denom[z] = 1.0
            else:
                t1_denom = t1

            # we don't want to use nan
            # (occassionally in cam data - often in ocn)
            m_t2 = np.ma.masked_invalid(t2).compressed()
            m_t1 = np.ma.masked_invalid(t1).compressed()

            if z.size > 0:
                m_t1_denom = np.ma.masked_invalid(t1_denom).compressed()
            else:
                m_t1_denom = m_t1

            m_tt = m_t1 - m_t2
            m_tt = m_tt / m_t1_denom

            max_spre = np.max(abs(m_tt))
            self._max_spatial_rel_error = max_spre

        return self._max_spatial_rel_error

    @property
    def ssim_value(self):
        """
        We compute the SSIM (structural similarity index) on the visualization of the spatial data.
        """

        import tempfile

        import skimage.io
        import skimage.metrics
        from skimage.metrics import structural_similarity as ssim

        if not self._is_memoized('_ssim_value'):

            # Prevent showing stuff
            backend_ = mpl.get_backend()
            mpl.use('Agg')

            d1 = self._metrics1.get_metric('ds')
            d2 = self._metrics2.get_metric('ds')
            lat1 = d1[self._metrics1._lat_coord_name]
            lat2 = d2[self._metrics2._lat_coord_name]
            lon1 = d1[self._metrics1._lon_coord_name]
            lon2 = d2[self._metrics2._lon_coord_name]

            latdim = d1.cf[self._metrics1._lon_coord_name].ndim
            central = 0.0  # might make this a parameter later
            if latdim == 2:  # probably pop
                central = 300.0
            # make periodic
            if latdim == 2:
                cy_lon1 = np.hstack((lon1, lon1[:, 0:1]))
                cy_lon2 = np.hstack((lon2, lon2[:, 0:1]))

                cy_lat1 = np.hstack((lat1, lat1[:, 0:1]))
                cy_lat2 = np.hstack((lat2, lat2[:, 0:1]))

                cy1 = add_cyclic_point(d1)
                cy2 = add_cyclic_point(d2)

            else:  # 1d
                cy1, cy_lon1 = add_cyclic_point(d1, coord=lon1)
                cy2, cy_lon2 = add_cyclic_point(d2, coord=lon2)
                cy_lat1 = lat1
                cy_lat2 = lat2

            no_inf_d1 = np.nan_to_num(cy1, nan=np.nan)
            no_inf_d2 = np.nan_to_num(cy2, nan=np.nan)

            # is it 3D? must do each level
            if self._metrics1._vert_dim_name is not None:
                vname = self._metrics1._vert_dim_name
                nlevels = self._metrics1.get_metric('ds').sizes[vname]
            else:
                nlevels = 1

            # print("nlevels = ", nlevels)

            color_min = min(
                np.min(d1.where(d1 != -np.inf)).values.min(),
                np.min(d2.where(d2 != -np.inf)).values.min(),
            )
            color_max = max(
                np.max(d1.where(d1 != np.inf)).values.max(),
                np.max(d2.where(d2 != np.inf)).values.max(),
            )

            mymap = copy.copy(plt.cm.get_cmap('coolwarm'))
            mymap.set_under(color='black')
            mymap.set_over(color='white')
            mymap.set_bad(alpha=0)

            ssim_levs = np.zeros(nlevels)

            for this_lev in range(nlevels):

                with tempfile.TemporaryDirectory() as tmpdirname:
                    filename_1, filename_2 = (
                        f'{tmpdirname}/t_ssim1.png',
                        f'{tmpdirname}/t_ssim2.png',
                    )

                    if nlevels == 1:
                        this_no_inf_d1 = no_inf_d1
                        this_no_inf_d2 = no_inf_d2
                    else:
                        # this assumes time level is first (TO DO: verify)
                        this_no_inf_d1 = no_inf_d1[this_lev, :, :]
                        this_no_inf_d2 = no_inf_d2[this_lev, :, :]

                    fig = plt.figure(dpi=300, figsize=(9, 2.5))

                    ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=central))
                    ax1.set_facecolor('#39ff14')

                    ax1.pcolormesh(
                        cy_lon1,
                        cy_lat1,
                        this_no_inf_d1,
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
                    plt.savefig(filename_1, bbox_inches=extent1, transparent=True, pad_inches=0)
                    ax1.axis('on')

                    ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=central))
                    ax2.set_facecolor('#39ff14')

                    ax2.pcolormesh(
                        cy_lon2,
                        cy_lat2,
                        this_no_inf_d2,
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
                    plt.savefig(filename_2, bbox_inches=extent2, transparent=True, pad_inches=0)

                    ax2.axis('on')

                    img1 = skimage.io.imread(filename_1)
                    img2 = skimage.io.imread(filename_2)
                    # scikit is adding an alpha channel for some reason - get rid of it
                    img1 = img1[:, :, :3]
                    img2 = img2[:, :, :3]

                    # s = ssim(img1, img2, multichannel=True)
                    # the following version closer to matlab version (and orig ssim paper)
                    s = ssim(
                        img1,
                        img2,
                        multichannel=True,
                        gaussian_weights=True,
                        use_sample_covariance=False,
                    )
                    # print(s)
                    ssim_levs[this_lev] = s
                    plt.close(fig)

            return_ssim = ssim_levs.min()

            # Reset backend
            mpl.use(backend_)

            self._ssim_value = return_ssim

        return self._ssim_value

    @property
    def ssim_value_fp(self):
        """
        We compute the SSIM (structural similarity index) on the spatial data - using the
        data itself (we do not create an image).

        Here we scale from [0,1] - then quantize to 256 bins

        """
        from math import exp, pi, sqrt

        from skimage.util import crop

        if not self._is_memoized('_ssim_value_fp'):

            # if this is a 3D variable, we will do each level seperately
            if self._metrics1._vert_dim_name is not None:
                vname = self._metrics1._vert_dim_name
                nlevels = self._metrics1.get_metric('ds').sizes[vname]
                # print("nlevels = ", nlevels)
            else:
                nlevels = 1

            ssim_levs = np.zeros(nlevels)

            for this_lev in range(nlevels):
                if nlevels == 1:
                    a1 = self._metrics1.get_metric('ds').data
                    a2 = self._metrics2.get_metric('ds').data
                else:
                    a1 = self._metrics1.get_metric('ds').isel({vname: this_lev}).data
                    a2 = self._metrics2.get_metric('ds').isel({vname: this_lev}).data

                if dask.is_dask_collection(a1):
                    a1 = a1.compute()
                if dask.is_dask_collection(a2):
                    a2 = a2.compute()

                # re-scale  to [0,1] - if not constant
                smin = min(np.nanmin(a1), np.nanmin(a2))
                smax = max(np.nanmax(a1), np.nanmax(a2))
                r = smax - smin
                if r == 0.0:  # scale by smax if fiels is a constant (and smax != 0)
                    if smax == 0.0:
                        sc_a1 = a1
                        sc_a2 = a2
                    else:
                        sc_a1 = a1 / smax
                        sc_a2 = a2 / smax
                else:
                    sc_a1 = (a1 - smin) / r
                    sc_a2 = (a2 - smin) / r

                # now quantize to 256 bins
                sc_a1 = np.round(sc_a1 * 255) / 255
                sc_a2 = np.round(sc_a2 * 255) / 255

                # gaussian filter
                n = 11  # recommended window size
                k = 5
                # extent
                sigma = 1.5

                X = sc_a1.shape[0]
                Y = sc_a2.shape[1]

                g_w = np.array(self._oned_gauss(n, sigma))
                # 2D gauss weights
                gg_w = np.outer(g_w, g_w)

                # init ssim matrix
                ssim_mat = np.zeros_like(sc_a1)

                my_eps = 1.0e-15

                # DATA LOOP
                # go through 2D arrays - each grid point x0, y0  has
                # a 2D window [x0 - k, x0+k]  [y0 - k, y0 + k]
                for i in range(X):

                    # don't go over boundaries
                    imin = max(0, i - k)
                    imax = min(X - 1, i + k)

                    for j in range(Y):

                        if np.isnan(sc_a1[i, j]):
                            # SKIP IF gridpoint is nan
                            ssim_mat[i, j] = np.nan
                            continue

                        jmin = max(0, j - k)
                        jmax = min(Y - 1, j + k)

                        # WINDOW CALC
                        a1_win = sc_a1[imin : imax + 1, jmin : jmax + 1]
                        a2_win = sc_a2[imin : imax + 1, jmin : jmax + 1]

                        # if window is by boundary, then it is not 11x11 and we must adjust weights also
                        if min(a1_win.shape) < n:
                            Wt = gg_w[
                                imin + k - i : imax + k - i + 1, jmin + k - j : jmax + k - j + 1
                            ]
                        else:
                            Wt = gg_w

                        # weighted means (TO DO: what if indices are not the same)
                        indices1 = ~np.isnan(a1_win)
                        indices2 = ~np.isnan(a2_win)

                        if not np.all(indices1 == indices2):
                            print('SSIM ERROR: indices are not the same!')

                        a1_mu = np.average(a1_win[indices1], weights=Wt[indices1])
                        a2_mu = np.average(a2_win[indices2], weights=Wt[indices2])

                        # weighted std squared (variance)
                        a1_std_sq = np.average(
                            (a1_win[indices1] - a1_mu) ** 2, weights=Wt[indices1]
                        )
                        a2_std_sq = np.average(
                            (a2_win[indices2] - a2_mu) ** 2, weights=Wt[indices2]
                        )

                        # cov of a1 and a2
                        a1a2_cov = np.average(
                            (a1_win[indices1] - a1_mu) * (a2_win[indices2] - a2_mu),
                            weights=Wt[indices1],
                        )

                        # SSIM for this window
                        # first term
                        ssim_t1 = 2 * a1_mu * a2_mu
                        ssim_b1 = a1_mu * a1_mu + a2_mu * a2_mu
                        C1 = my_eps

                        ssim_t1 = ssim_t1 + C1
                        ssim_b1 = ssim_b1 + C1

                        # second term
                        ssim_t2 = 2 * a1a2_cov
                        ssim_b2 = a1_std_sq + a2_std_sq
                        C2 = C3 = my_eps

                        ssim_t2 = ssim_t2 + C3
                        ssim_b2 = ssim_b2 + C2

                        ssim_1 = ssim_t1 / ssim_b1
                        ssim_2 = ssim_t2 / ssim_b2
                        ssim_mat[i, j] = ssim_1 * ssim_2

                mean_ssim = np.nanmean(ssim_mat)
                # to ignore edge effects (as in skimage)
                # mean_ssim = np.nanmean(crop(ssim_mat, k))
                ssim_levs[this_lev] = mean_ssim

            return_ssim = ssim_levs.min()
            self._ssim_value_fp = return_ssim

        return self._ssim_value_fp

    @property
    def ssim_value_fp_old(self):
        """To mimc what zchecker does - the ssim on the fp data with
        original constants and no scaling. This will return Nan on POP data.
        """

        import numpy as np
        from skimage.metrics import structural_similarity as ssim

        if not self._is_memoized('_ssim_value_fp_old'):

            # if this is a 3D variable, we will just do level 0 for now...
            # (consider doing each level seperately)
            if self._metrics1._vert_dim_name is not None:
                vname = self._metrics1._vert_dim_name
                a1 = self._metrics1.get_metric('ds').isel({vname: 0}).data
                a2 = self._metrics2.get_metric('ds').isel({vname: 0}).data
            else:
                a1 = self._metrics1.get_metric('ds').data
                a2 = self._metrics2.get_metric('ds').data

            if dask.is_dask_collection(a1):
                a1 = a1.compute()
            if dask.is_dask_collection(a2):
                a2 = a2.compute()

            maxr = max(a1.max(), a2.max())
            minr = min(a1.min(), a2.min())
            myrange = maxr - minr
            mean_ssim = ssim(
                a1,
                a2,
                multichannel=False,
                data_range=myrange,
                gaussian_weights=True,
                use_sample_covariance=False,
            )

            self._ssim_value_fp_old = mean_ssim

        return self._ssim_value_fp_old

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
            if name == 'max_spatial_rel_error':
                return self.max_spatial_rel_error
            if name == 'ssim':
                return self.ssim_value
            if name == 'ssim_fp':
                return self.ssim_value_fp
            if name == 'ssim_fp_old':
                return self.ssim_value_fp_old
            raise ValueError(f'there is no metric with the name: {name}.')
        else:
            raise TypeError('name must be a string.')
