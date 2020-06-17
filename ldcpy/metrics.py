from typing import Optional

import numpy as np
import scipy.stats as ss
import xarray as xr


class DatasetMetrics(object):
    """
    This class contains metrics for each point of a dataset, and a method to access these metrics.
    """

    def __init__(self, ds: xr.DataArray) -> None:
        if isinstance(ds, xr.DataArray):
            # Datasets
            self._ds = ds

            # Variables
            self._ns_con_var_full = None
            self._ew_con_var_full = None
            self._is_positive_full = None
            self._is_negative_full = None

        else:
            raise TypeError(
                f'dataset must be of type xarray.DataArray. Type(observed): {str(type(ds))}'
            )

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    @property
    def ns_con_var_full(self) -> np.ndarray:
        """
        The North-South contrast variance at each point in the dataset
        """
        if not self._is_memoized('_ns_con_var_full'):
            self._ns_con_var_full = self._con_var('ns', self._ds)

        return self._ns_con_var_full

    @property
    def ew_con_var_full(self) -> np.ndarray:
        """
        The East-West contrast variance at each point in the dataset
        """
        if not self._is_memoized('_ew_con_var_full'):
            self._ew_con_var_full = self._con_var('ew', self._ds)

        return self._ew_con_var_full

    @property
    def is_positive_full(self) -> np.ndarray:
        """
        returns 1 if the data point is positive, 0 if negative
        """
        if not self._is_memoized('_is_positive_full'):
            self._is_positive_full = self._ds > 0

        return self._is_positive_full

    @property
    def is_negative_full(self) -> np.ndarray:
        """
        returns 1 if the data point is negative, 0 if positive
        """
        if not self._is_memoized('_is_negative_full'):
            self._is_negative_full = self._ds < 0

        return self._is_negative_full

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
        con_var = xr.ufuncs.square((o_1 - o_2))
        return con_var

    def get_full_metric(self, name: str):
        """
        Gets a metric at every data point of the dataset

        Parameters:
        ===========
        name -- string
            the name of the metric (must be identical to a property name)

        Returns
        =======
        out -- xarray.DataArray
            a DataArray of the same size and dimensions the original dataarray, with metrics at each data point
        """
        if isinstance(name, str):
            if name == 'ns_con_var_full':
                return self.ns_con_var_full
            if name == 'ew_con_var_full':
                return self.ew_con_var_full
            if name == 'is_positive_full':
                return self._is_positive_full
            if name == 'is_negative_full':
                return self._is_negative_full
            raise ValueError(f'there are no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class AggregateMetrics(DatasetMetrics):
    """
    This class contains metrics for each point of a dataset after aggregating across one or more dimensions, and a method to access these metrics.
    """

    def __init__(self, ds: xr.DataArray, aggregate_dims: list, grouping: Optional[str] = None):
        DatasetMetrics.__init__(self, ds)

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
        self._grouping = grouping

        self._frame_size = 1
        for dim in aggregate_dims:
            self._frame_size *= int(self._ds.sizes[dim])

    @property
    def ns_con_var(self) -> np.ndarray:
        """
        The North-South Contrast Variance averaged along the aggregate dimensions
        """
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var = self._con_var('ns', self._ds).mean(self._agg_dims)

        return self._ns_con_var

    @property
    def ew_con_var(self) -> np.ndarray:
        """
        The East-West Contrast Variance averaged along the aggregate dimensions
        """
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var = self._con_var('ew', self._ds).mean(self._agg_dims)

        return self._ew_con_var

    @property
    def mean(self) -> np.ndarray:
        """
        The mean along the aggregate dimensions
        """
        if not self._is_memoized('_mean'):
            self._mean = self._ds.mean(self._agg_dims)

        return self._mean

    @property
    def mean_abs(self) -> np.ndarray:
        """
        The absolute value of the mean along the aggregate dimensions
        """
        if not self._is_memoized('_mean_abs'):
            self._mean_abs = abs(self.mean)

        return self._mean_abs

    @property
    def std(self) -> np.ndarray:
        """
        The standard deviation averaged along the aggregate dimensions
        """
        if not self._is_memoized('_std'):
            self._std = self._ds.std(self._agg_dims, ddof=1)

        return self._std

    @property
    def prob_positive(self) -> np.ndarray:
        """
        The probability that a point is positive
        """
        if not self._is_memoized('_prob_positive'):
            self._prob_positive = (self.is_positive_full.sum(self._agg_dims)) / (self._frame_size)
        return self._prob_positive

    @property
    def prob_negative(self) -> np.ndarray:
        """
        The probability that a point is negative
        """
        if not self._is_memoized('_prob_negative'):
            self._prob_negative = self.is_negative_full.sum(self._agg_dims) / self._frame_size
        return self._prob_negative

    @property
    def odds_positive(self) -> np.ndarray:
        """
        The odds that a point is positive: prob_positive/(1-prob_positive)
        """
        if not self._is_memoized('_odds_positive'):
            self._odds_positive = self.prob_positive / (1 - self.prob_positive)
        return self._odds_positive

    @property
    def zscore(self) -> np.ndarray:
        """
        The z-score of a point averaged along the aggregate dimensions under the null hypothesis that the true mean is zero.
        """
        if not self._is_memoized('_zscore'):
            self._zscore = np.divide(self.mean, self.std / np.sqrt(self._ds.sizes['time']))

        return self._zscore

    @property
    def mae_max(self) -> xr.DataArray:
        """
        The maximum mean absolute error at the point averaged along the aggregate dimensions.
        """
        if not self._is_memoized('_mae_day_max'):
            self._test = abs(self._ds.groupby(self._grouping).mean(dim=self._agg_dims))
            # Would be great to replace the code below with a single call to _test.idxmax() once idxmax is in a stable release
            if self._grouping == 'time.dayofyear':
                self._mae_max = xr.DataArray(
                    self._test.isel(dayofyear=self._test.argmax(dim='dayofyear'))
                    .coords.variables.mapping['dayofyear']
                    .data,
                    dims=['lat', 'lon'],
                )
            if self._grouping == 'time.month':
                self._mae_max = xr.DataArray(
                    self._test.isel(month=self._test.argmax(dim='month'))
                    .coords.variables.mapping['month']
                    .data,
                    dims=['lat', 'lon'],
                )
            if self._grouping == 'time.year':
                self._mae_max = xr.DataArray(
                    self._test.isel(year=self._test.argmax(dim='year'))
                    .coords.variables.mapping['year']
                    .data,
                    dims=['lat', 'lon'],
                )

        return self._mae_max

    @property
    def lag1(self) -> xr.DataArray:
        """
        The deseasonalized lag-1 value by day of year
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
            self._lag1 = xr.ufuncs.square((o_1 - o_2))

        return self._lag1

    @property
    def corr_lag1(self) -> xr.DataArray:
        """
        The deseasonalized lag-1 correlation at each point by day of year
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

        return self._corr_lag1

    def get_metric(self, name: str):
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
            if name == 'corr_lag1':
                return self.corr_lag1
            raise ValueError(f'there is no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class OverallMetrics(AggregateMetrics):
    """
    This class contains metrics on the overall dataset
    """

    def __init__(self, ds: xr.DataArray, aggregate_dims: list):
        AggregateMetrics.__init__(self, ds, aggregate_dims)

        self._zscore_cutoff = None
        self._zscore_percent_significant = None

    @property
    def zscore_cutoff(self) -> np.ndarray:
        """
        The Z-Score cutoff for a point to be considered significant
        """
        if not self._is_memoized('_zscore_cutoff'):
            pvals = 2 * (1 - ss.norm.cdf(np.abs(self.zscore)))
            sorted_pvals = np.sort(pvals).flatten()
            fdr_zscore = 0.01
            p = np.argwhere(sorted_pvals <= fdr_zscore * np.arange(1, pvals.size + 1) / pvals.size)
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
        """
        if not self._is_memoized('_zscore_percent_significant'):
            pvals = 2 * (1 - ss.norm.cdf(np.abs(self.zscore)))
            sorted_pvals = np.sort(pvals).flatten()
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

    def get_overall_metric(self, name: str):
        """
        Gets a single metric on the dataset

        Parameters:
        ===========
        name -- string
            the name of the metric (must be identical to a property name)

        Returns
        =======
        out -- float32
        """
        if isinstance(name, str):
            if name == 'zscore_cutoff':
                return self.zscore_cutoff
            if name == 'zscore_percent_significant':
                return self.zscore_percent_significant
            raise ValueError(f'there is no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')
