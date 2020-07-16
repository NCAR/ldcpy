from typing import Optional

import numpy as np
import xarray as xr
from scipy import stats as ss


class DatasetMetrics(object):
    """
    This class contains metrics for each point of a dataset after aggregating across one or more dimensions, and a method to access these metrics.
    """

    def __init__(
        self, ds: xr.DataArray, aggregate_dims: list,
    ):
        self._ds = ds if (ds.dtype == np.float64) else ds.astype(np.float64)
        # For some reason, casting to float64 removes all attrs from the dataset
        self._ds.attrs = ds.attrs

        # array metrics
        self._ns_con_var = None
        self._ew_con_var = None
        self._mean = None
        self._std = None
        self._prob_positive = None
        self._odds_positive = None
        self._prob_negative = None
        self._zscore = None
        self._mae_max = None
        self._corr_lag1 = None
        self._lag1 = None
        self._agg_dims = aggregate_dims
        self._quantile_value = None
        self._mean_squared = None
        self._root_mean_squared = None
        self._sum = None
        self._sum_squared = None
        self._variance = None
        self._quantile = 0.5
        self._max_abs = None
        self._min_abs = None
        self._d_range = None
        self._min_val = None
        self._max_val = None

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
                    [dataset.tail({'lon': lon_length - 1}), dataset.head({'lon': 1})], dim='lon',
                ),
                join='override',
            )
        # con_var = xr.ufuncs.square((o_1 - o_2))
        con_var = np.square((o_1 - o_2))
        return con_var

    @property
    def ns_con_var(self) -> np.ndarray:
        """
        The North-South Contrast Variance averaged along the aggregate dimensions
        """
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var = self._con_var('ns', self._ds).mean(self._agg_dims)
            self._ns_con_var.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._ns_con_var.attrs['units'] = f'{self._ds.units}^2'

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
                self._ew_con_var.attrs['units'] = f'{self._ds.units}^2'

        return self._ew_con_var

    @property
    def mean(self) -> np.ndarray:
        """
        The mean along the aggregate dimensions
        """
        if not self._is_memoized('_mean'):
            self._mean = self._ds.mean(self._agg_dims)
            self._mean.attrs = self._ds.attrs

        return self._mean

    @property
    def mean_abs(self) -> np.ndarray:
        """
        The mean of the absolute errors along the aggregate dimensions
        """
        if not self._is_memoized('_mean_abs'):
            self._mean_abs = abs(self._ds).mean(self._agg_dims)
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
            self._mean_abs.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._mean_abs.attrs['units'] = f'{self._ds.units}^2'

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
            self._sum = self._ds.sum(dim=self._agg_dims)
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
                self._sum_squared.attrs['units'] = f'{self._ds.units}^2'

        return self._sum_squared

    @property
    def std(self) -> np.ndarray:
        """
        The standard deviation along the aggregate dimensions
        """
        if not self._is_memoized('_std'):
            self._std = self._ds.std(self._agg_dims)
            self._std.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._std.attrs['units'] = ''

        return self._std

    @property
    def variance(self) -> np.ndarray:
        """
        The variance along the aggregate dimensions
        """
        if not self._is_memoized('_variance'):
            self._variance = self._ds.var(self._agg_dims)
            self._variance.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._variance.attrs['units'] = f'{self._ds.units}^2'

        return self._variance

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
        The odds that a point is positive: prob_positive/(1-prob_positive)
        """
        if not self._is_memoized('_odds_positive'):
            self._odds_positive = self.prob_positive / (1 - self.prob_positive)
            self._odds_positive.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._odds_positive.attrs['units'] = ''
        return self._odds_positive

    @property
    def zscore(self) -> np.ndarray:
        """
        The z-score of a point averaged along the aggregate dimensions under the null hypothesis that the true mean is zero.
        """
        if not self._is_memoized('_zscore'):
            self._zscore = np.divide(self.mean, self.std / np.sqrt(self._ds.sizes['time']))
            self._zscore.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._zscore.attrs['units'] = ''

        return self._zscore

    @property
    def mae_max(self) -> xr.DataArray:
        """
        The maximum mean absolute error at the point averaged along the aggregate dimensions.
        TODO: There is a bug hiding in this code, plotting the values does not work correctly. Waiting on xarray 0.15.2 when ds.idxmax() will available (use self._test.idxmax())
        """
        if not self._is_memoized('_mae_day_max'):
            self._mae_max = 0
            # self._test = abs(self._ds.groupby(self._grouping).mean(dim=self._agg_dims))
            # # Would be great to replace the code below with a single call to _test.idxmax() once idxmax is in a stable release
            # if self._grouping == 'time.dayofyear':
            #     self._mae_max = xr.DataArray(
            #         self._test.isel(dayofyear=self._test.argmax(dim='dayofyear'))
            #         .coords.variables.mapping['dayofyear']
            #         .data,
            #         dims=['lat', 'lon'],
            #     )
            # if self._grouping == 'time.month':
            #     self._mae_max = xr.DataArray(
            #         self._test.isel(month=self._test.argmax(dim='month'))
            #         .coords.variables.mapping['month']
            #         .data,
            #         dims=['lat', 'lon'],
            #     )
            # if self._grouping == 'time.year':
            #     self._mae_max = xr.DataArray(
            #         self._test.isel(year=self._test.argmax(dim='year'))
            #         .coords.variables.mapping['year']
            #         .data,
            #         dims=['lat', 'lon'],
            #     )
            self._mae_max.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._mae_max.attrs['units'] = f'{self._ds.units}'

        return self._mae_max

    @property
    def quantile(self):
        return self._quantile

    @quantile.setter
    def quantile(self, q):
        self._quantile = q

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
            self._max_abs = abs(self._ds).max()
            self._max_abs.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._max_abs.attrs['units'] = f'{self._ds.units}'

        return self._max_abs

    @property
    def min_abs(self) -> xr.DataArray:
        if not self._is_memoized('_min_abs'):
            self._min_abs = abs(self._ds).min()
            self._min_abs.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._min_abs.attrs['units'] = f'{self._ds.units}'

        return self._min_abs

    @property
    def max_val(self) -> xr.DataArray:
        if not self._is_memoized('_max_val'):
            self._max_val = self._ds.max()
            self._max_val.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._max_val.attrs['units'] = f'{self._ds.units}'

        return self._max_val

    @property
    def min_val(self) -> xr.DataArray:
        if not self._is_memoized('_min_val'):
            self._min_val = self._ds.min()
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
        The deseasonalized lag-1 value by day of year
        TODO: This metric currently returns a lat-lon array regardless of aggregate dimensions, so can only be used in a spatial plot.
        """
        if not self._is_memoized('_lag1'):
            self._deseas_resid = self._ds.groupby('time.dayofyear') - self._ds.groupby(
                'time.dayofyear'
            ).mean(dim='time')

            time_length = self._ds.sizes['time']
            o_1, o_2 = xr.align(
                self._deseas_resid.head({'time': time_length - 1}),
                self._deseas_resid.tail({'time': time_length - 1}),
                join='override',
            )
            self._lag1 = np.square((o_1 - o_2))
            self._lag1.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._lag1.attrs['units'] = ''

        return self._lag1

    @property
    def corr_lag1(self) -> xr.DataArray:
        """
        The deseasonalized lag-1 correlation at each point by day of year
        TODO: This metric currently returns a lat-lon array regardless of aggregate dimensions, so can only be used in a spatial plot.
        """
        if not self._is_memoized('_corr_lag1'):
            time_length = self._ds.sizes['time']
            l_1, l_2 = xr.align(
                self.lag1.head({'time': time_length - 2}),
                self.lag1.tail({'time': time_length - 2}),
                join='override',
            )
            self._corr_lag1 = np.multiply(l_1, l_2).sum(dim='time') / np.multiply(
                self._lag1, self._lag1
            ).sum(dim='time')
            self._corr_lag1.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._corr_lag1.attrs['units'] = ''

        return self._corr_lag1

    @property
    def zscore_cutoff(self) -> np.ndarray:
        """
        The Z-Score cutoff for a point to be considered significant
        TODO: Some single-value properties (liek this one) cannot be used in either spatial or time-series plots, there needs to be a way to specify which these are.
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
    def zscore_percent_significant(self) -> np.ndarray:
        """
        The percent of points where the zscore is considered significant
        TODO: Some single-value properties (liek this one) cannot be used in either spatial or time-series plots, there needs to be a way to specify which these are.
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

    def get_metric(self, name: str, q: Optional[int] = 0.5):
        """
        Gets a metric aggregated across one or more dimensions of the dataset

        Parameters:
        ===========
        name -- string
            the name of the metric (must be identical to a property name)

        Returns
        =======
        out -- xarray.DataArray
            a DataArray of the same size and dimensions the original dataarray, minus those dimensions that were aggregated across.
        """
        if isinstance(name, str):
            if name == 'ns_con_var':
                return self.ns_con_var
            if name == 'ew_con_var':
                return self.ew_con_var
            if name == 'mean':
                return self.mean
            if name == 'std':
                return self.std
            if name == 'variance':
                return self.variance
            if name == 'prob_positive':
                return self.prob_positive
            if name == 'prob_negative':
                return self.prob_negative
            if name == 'odds_positive':
                return self.odds_positive
            if name == 'zscore':
                return self.zscore
            if name == 'mae_max':
                return self.mae_max
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
            if name == 'corr_lag1':
                return self.corr_lag1
            if name == 'quantile':
                self.quantile = q
                return self.quantile_value
            if name == 'lag1':
                return self.lag1
            if name == 'max_abs':
                return self.max_abs
            if name == 'min_abs':
                return self.min_abs
            if name == 'max_val':
                return self.max_val
            if name == 'min_val':
                return self.min_val
            if name == 'range':
                return self.dyn_range
            if name == 'ds':
                return self._ds
            raise ValueError(f'there is no metric with the name: {name}.')
        else:
            raise TypeError('name must be a string.')

    def get_single_metric(self, name: str):
        """
        Gets a metric consisting of a single float value

        Parameters:
        ===========
        name -- string
            the name of the metric (must be identical to a property name)

        Returns
        =======
        out -- float
            the metric value
        """
        if isinstance(name, str):
            if name == 'zscore_cutoff':
                return self.zscore_cutoff
            if name == 'zscore_percent_significant':
                return self.zscore_percent_significant
            raise ValueError(f'there is no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class DiffMetrics(object):
    """
    This class contains metrics on the overall dataset that require more than one input dataset to compute
    """

    def __init__(
        self, ds1: xr.DataArray, ds2: xr.DataArray, aggregate_dims: Optional[list] = None,
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
        if not self._is_memoized('_ks_p_value'):
            self._ks_p_value = np.asanyarray(ss.ks_2samp(np.ravel(self._ds2), np.ravel(self._ds1)))
        return self._ks_p_value

    @property
    def pearson_correlation_coefficient(self):
        """
        returns the pearson correlation coefficient between the two datasets
        """
        if not self._is_memoized('_pearson_correlation_coefficient'):
            self._pcc = (
                self.covariance
                / self._metrics1.get_metric('std')
                / self._metrics2.get_metric('std')
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

    def get_diff_metric(self, name: str):
        """
        Gets a metric on the dataset that requires more than one input dataset

        Parameters:
        ===========
        name -- string
            the name of the metric (must be identical to a property name)

        Returns
        =======
        out -- float32
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
            raise ValueError(f'there is no metric with the name: {name}.')
        else:
            raise TypeError('name must be a string.')
