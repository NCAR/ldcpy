import collections
import typing

import dask
import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr
import xrft

xr.set_options(keep_attrs=True)


@xr.register_dataarray_accessor('ldc')
class MetricsAccessor:
    """
    This class contains metrics for each point of a dataset after aggregating across one or more dimensions, and a method to access these metrics.
    """

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        keys = [
            'ns_con_var',
            'ew_con_var',
            'mean_',
            'mean_abs',
            'standard_deviation',
            'prob_positive',
            'odds_positive',
            'prob_negative',
            'zscore',
            'mae_day_max',
            'lag1',
            'lag1_first_difference',
            'quantile_value',
            'mean_squared',
            'root_mean_squared',
            'sum_',
            'sum_squared',
            'variance',
            'pooled_variance',
            'pooled_variance_ratio',
            'standardized_mean',
            'max_abs',
            'max_abs',
            'min_abs',
            'dyn_range',
            'min_val',
            'max_val',
            'annual_harmonic_relative_ratio',
            'annual_harmonic_relative_ratio_pct_sig',
            'zscore_cutoff',
            'zscore_percent_significant',
        ]
        self._metrics = pd.Series(collections.OrderedDict.fromkeys(sorted(keys)))
        self._attrs = self._obj.attrs.copy()
        self._aggregate_dims = None
        self._is_computed = False

    def __call__(
        self,
        aggregate_dims: typing.Union[typing.List, typing.Tuple],
        grouping: typing.Union[typing.List, typing.Tuple] = None,
        ddof: int = 1,
        q: float = 0.5,
        spre_tol: float = 1.0e-4,
        time_dim_name: str = 'time',
        lat_dim_name: str = 'lat',
        lon_dim_name: str = 'lon',
    ):
        """
        [summary]

        Parameters
        ----------
        aggregate_dims : typing.Union[typing.List, typing.Tuple]
            [description]
        grouping : typing.Union[typing.List, typing.Tuple], optional
            [description], by default None
        ddof : int, optional
            [description], by default 1
        q : float, optional
            [description], by default 0.5
        spre_tol : float, optional
            [description], by default 1.0e-4
        time_dim_name : str, optional
            [description], by default 'time'
        lat_dim_name : str, optional
            [description], by default 'lat'
        lon_dim_name : str, optional
            [description], by default 'lon'

        Returns
        -------
        [type]
            [description]
        """
        self._aggregate_dims = aggregate_dims
        self._frame_size = 1
        if self._aggregate_dims:
            for dim in self._aggregate_dims:
                self._frame_size *= int(self._obj.sizes[dim])
        self._grouping = grouping
        self._ddof = ddof
        self._q = q
        self._spre_tol = spre_tol
        self._time_dim_name = time_dim_name
        self._lat_dim_name = lat_dim_name
        self._lon_dim_name = lon_dim_name
        return self

    def _con_var(self, con_type):
        if con_type == 'ns':
            lat_length = self._obj.sizes[self._lat_dim_name]
            o_1, o_2 = xr.align(
                self._obj.head({self._lat_dim_name: lat_length - 1}),
                self._obj.tail({self._lat_dim_name: lat_length - 1}),
                join='override',
            )
        elif con_type == 'ew':
            lon_length = self._obj.sizes[self._lon_dim_name]
            o_1, o_2 = xr.align(
                self._obj,
                xr.concat(
                    [
                        self._obj.tail({self._lon_dim_name: lon_length - 1}),
                        self._obj.head({self._lon_dim_name: 1}),
                    ],
                    dim=self._lon_dim_name,
                ),
                join='override',
            )

        con_var = np.square((o_1 - o_2)).mean(self._aggregate_dims)
        return con_var

    def _standardized_mean(self):
        if self._grouping is None:
            standardized_mean = (self._metrics.mean_ - self._obj.mean()) / self._obj.std(ddof=1)
        else:
            m = self._metrics.mean_.groupby(self._grouping).mean()
            standardized_mean = (m - m.mean()) / m.std(ddof=1)

        return standardized_mean

    def _odds_positive(self):
        if self._grouping is None:
            odds_positive = self._metrics.prob_positive / (1 - self._metrics.prob_positive)
        else:
            op = self._metrics.prob_positive.groupby(self._grouping).mean()
            odds_positive = op / (1 - op)
        return odds_positive

    def _mae_day_max(self):
        key = f'{self._time_dim_name}.dayofyear'
        mae_day_max = np.abs(self._obj).groupby(key).mean().idxmax(dim='dayofyear')
        return mae_day_max

    def _lag_components(self):
        key = f'{self._time_dim_name}.dayofyear'
        grouped = self._obj.groupby(key)
        deseas_resid = grouped - grouped.mean(dim=self._time_dim_name)
        time_length = deseas_resid.sizes[self._time_dim_name]
        current_val = deseas_resid.head({self._time_dim_name: time_length - 1})
        next_val = deseas_resid.shift({self._time_dim_name: -1}).head(
            {self._time_dim_name: time_length - 1}
        )
        return current_val, next_val, time_length

    def _lag1_first_difference(self, current_val, next_val, time_length):
        """
        The deseasonalized lag-1 autocorrelation value of the first difference of the data by day of year
        NOTE: This metric returns an array of spatial values as the data set regardless of aggregate dimensions,
        so can only be plotted in a spatial plot.
        """
        first_difference = next_val - current_val
        first_difference_current = first_difference.head({self._time_dim_name: time_length - 1})
        first_difference_next = first_difference.shift({self._time_dim_name: -1}).head(
            {self._time_dim_name: time_length - 1}
        )
        num = (first_difference_current * first_difference_next).sum(
            dim=self._time_dim_name, skipna=True
        )
        denom = first_difference_current.dot(first_difference_current, dims=self._time_dim_name)
        lag1_first_difference = num / denom
        return lag1_first_difference

    def _annual_harmonic_relative_ratio(self):
        """
        The annual harmonic relative to the average periodogram value
        in a neighborhood of 50 frequencies around the annual frequency
        NOTE: This assumes the values along the "time" dimension are equally spaced.
        NOTE: This metric returns a lat-lon array regardless of aggregate dimensions, so can only be used in a spatial plot.
        """
        new_index = [i for i in range(0, self._obj[self._time_dim_name].size)]
        new_ds = self._obj.copy().assign_coords({self._time_dim_name: new_index})

        DF = xrft.dft(new_ds, dim=[self._time_dim_name], detrend='constant')
        S = np.real(DF * np.conj(DF) / self._obj.sizes[self._time_dim_name])
        a = int(self._obj.sizes[self._time_dim_name] / 2)
        b = int(self._obj.sizes[self._time_dim_name] / 365)
        S_annual = S.isel(freq_time=a + b)
        neighborhood = (a + b - 25, a + b + 25)
        S_mean = xr.concat(
            [
                S.isel(freq_time=slice(max(0, neighborhood[0]), a + b - 1)),
                S.isel(freq_time=slice(a + b + 1, neighborhood[1])),
            ],
            dim='freq_time',
        ).mean(dim='freq_time')
        ratio = S_annual / S_mean
        return ratio

    def _annual_harmonic_relative_ratio_pct_sig(self):
        pvals = 1 - scipy.stats.f.cdf(self._metrics.annual_harmonic_relative_ratio, 2, 100)
        sorted_pvals = np.sort(pvals)
        if len(sorted_pvals[sorted_pvals <= 0.01]) == 0:
            return 0
        sig_cutoff = scipy.stats.f.ppf(1 - max(sorted_pvals[sorted_pvals <= 0.01]), 2, 50)
        pct_sig = 100 * np.mean(self._metrics.annual_harmonic_relative_ratio > sig_cutoff)
        return pct_sig

    def _zscore_cutoff_and_percent_sig(self):
        # TODO: Find ways to Daskify this method
        pvals = 2 * (1 - scipy.stats.norm.cdf(np.abs(self._metrics.zscore)))
        if isinstance(pvals, np.float64):
            pvals = np.array(pvals)
        sorted_pvals = np.sort(pvals).flatten()

        fdr_szcore = 0.01
        p = np.argwhere(
            sorted_pvals <= fdr_szcore * np.arange(1, sorted_pvals.size + 1) / sorted_pvals.size
        )
        if len(p) > 0:
            pval_cutoff = sorted_pvals[p[len(p) - 1]]
        else:
            pval_cutoff = np.empty(0)

        if pval_cutoff.size > 0:
            zscore_cutoff = scipy.stats.norm.ppf(1 - pval_cutoff)
            sig_locs = np.argwhere(pvals <= pval_cutoff)
            percent_sig = 100.0 * np.size(sig_locs, 0) / pvals.size
        else:
            percent_sig = 0.0
            zscore_cutoff = np.nan

        return zscore_cutoff, percent_sig

    def _compute_metrics(self):
        self._metrics.ns_con_var = self._con_var('ew')
        self._metrics.ew_con_var = self._con_var('ns')

        self._metrics.pooled_variance = self._obj.var(self._aggregate_dims).mean()
        self._metrics.variance = self._obj.var(self._aggregate_dims, skipna=True)
        self._metrics.pooled_variance_ratio = self._metrics.variance / self._metrics.pooled_variance

        self._metrics.mean_ = self._obj.mean(self._aggregate_dims)
        self._metrics.mean_squared = self._metrics.mean_ ** 2
        self._metrics.mean_abs = np.abs(self._obj).mean(self._aggregate_dims, skipna=True)

        self._metrics.root_mean_squared = np.sqrt((self._obj ** 2).mean(self._aggregate_dims))

        self._metrics.sum_ = self._obj.sum(self._aggregate_dims)
        self._metrics.sum_squared = self._metrics.sum_ ** 2

        self._metrics.standard_deviation = self._obj.std(
            self._aggregate_dims, ddof=self._ddof, skipna=True
        )
        self._metrics.standardized_mean = self._standardized_mean()

        self._metrics.prob_positive = (self._obj > 0).sum(self._aggregate_dims) / self._frame_size
        self._metrics.prob_negative = (self._obj < 0).sum(self._aggregate_dims) / self._frame_size
        self._metrics.odds_positive = self._odds_positive()

        self._metrics.zscore = self._metrics.mean_ / (
            self._metrics.standard_deviation / np.sqrt(self._obj.sizes[self._time_dim_name])
        )
        self._metrics.mae_day_max = self._mae_day_max()

        self._metrics.quantile_value = self._obj.quantile(self._q, dim=self._aggregate_dims)

        abs_val = np.abs(self._obj)
        self._metrics.max_abs = abs_val.max(dim=self._aggregate_dims)
        self._metrics.min_abs = abs_val.min(dim=self._aggregate_dims)

        self._metrics.max_val = self._obj.max(dim=self._aggregate_dims)
        self._metrics.min_val = self._obj.min(dim=self._aggregate_dims)

        self._metrics.dyn_range = np.abs(self._obj.max() - self._obj.min())

        current_val, next_val, time_length = self._lag_components()
        self._metrics.lag1 = current_val.dot(next_val, dims=self._time_dim_name) / current_val.dot(
            current_val, dims=self._time_dim_name
        )
        self._metrics.lag1_first_difference = self._lag1_first_difference(
            current_val, next_val, time_length
        )

        self._metrics.annual_harmonic_relative_ratio = self._annual_harmonic_relative_ratio()
        (
            self._metrics.zscore_cutoff,
            self._metrics.zscore_percent_significant,
        ) = self._zscore_cutoff_and_percent_sig()

        self._metrics.annual_harmonic_relative_ratio_pct_sig = (
            self._annual_harmonic_relative_ratio_pct_sig()
        )

        if dask.is_dask_collection(self._obj):
            # Compute all metrics
            self._metrics = pd.Series(dask.compute(self._metrics.to_dict())[0])

        self._is_computed = True

    @property
    def metrics(self):
        if not self._is_computed:
            self._compute_metrics()
        return self._metrics
