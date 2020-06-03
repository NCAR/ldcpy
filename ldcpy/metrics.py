import numpy as np
import xarray as xr


class DatasetMetrics(object):
    def __init__(self, ds: xr.DataArray) -> None:
        if isinstance(ds, xr.DataArray):
            # Datasets
            self._ds = ds

            # Variables
            self._ns_con_var = None
            self._ew_con_var = None
            self._is_positive = None
            self._is_negative = None

        else:
            raise TypeError(
                f'dataset must be of type xarray.DataArray. Type(observed): {str(type(ds))}'
            )

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    @property
    def ns_con_var(self) -> np.ndarray:
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var = self._con_var('ns', self._ds)

        return self._ns_con_var

    @property
    def ew_con_var(self) -> np.ndarray:
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var = self._con_var('ew', self._ds)

        return self._ew_con_var

    @property
    def is_positive(self) -> np.ndarray:
        if not self._is_memoized('_is_positive'):
            self._is_positive = self._ds > 0

        return self._is_positive

    @property
    def is_negative(self) -> np.ndarray:
        if not self._is_memoized('_is_negative'):
            self._is_negative = self._ds < 0

        return self._is_negative

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

    def get_metric(self, name: str):
        if isinstance(name, str):
            if name == 'ns_con_var':
                return self.ns_con_var
            if name == 'ew_con_var':
                return self.ew_con_var
            if name == 'is_positive':
                return self._is_positive
            if name == 'is_negative':
                return self._is_negative
            raise ValueError(f'there are no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class SpatialMetrics(DatasetMetrics):
    def __init__(self, ds: xr.DataArray, aggregate_dims: list):
        DatasetMetrics.__init__(self, ds)

        self._ns_con_var_spatial = None
        self._ew_con_var_spatial = None
        self._mean_spatial = None
        self._std_spatial = None
        self._prob_positive_spatial = None
        self._odds_positive_spatial = None
        self._prob_negative_spatial = None
        self._zscore_spatial = None
        self._agg_dims = aggregate_dims

        self._frame_size = 1
        for dim in aggregate_dims:
            self._frame_size *= int(self._ds.sizes[dim])

    @property
    def ns_con_var_spatial(self) -> np.ndarray:
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var_spatial = self._con_var('ns', self._ds).mean(self._agg_dims)

        return self._ns_con_var_spatial

    @property
    def ew_con_var_spatial(self) -> np.ndarray:
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var_spatial = self._con_var('ew', self._ds).mean(self._agg_dims)

        return self._ew_con_var_spatial

    @property
    def mean_spatial(self) -> np.ndarray:
        if not self._is_memoized('_mean_spatial'):
            self._mean_spatial = self._ds.mean(self._agg_dims)

        return self._mean_spatial

    @property
    def std_spatial(self) -> np.ndarray:
        if not self._is_memoized('_std_spatial'):
            self._std_spatial = self._ds.std(self._agg_dims, ddof=1)

        return self._std_spatial

    @property
    def prob_positive_spatial(self) -> np.ndarray:
        if not self._is_memoized('_prob_positive_orig_spatial'):
            self._prob_positive_spatial = (self.is_positive.sum(self._agg_dims)) / (
                self._frame_size
            )
        return self._prob_positive_spatial

    @property
    def prob_negative_spatial(self) -> np.ndarray:
        if not self._is_memoized('_prob_negative_spatial'):
            self._prob_negative_spatial = (
                self.is_negative.sum(self._agg_dims) / self.is_negative.sizes['time']
            )
        return self._prob_negative_spatial

    @property
    def odds_positive_spatial(self) -> np.ndarray:
        if not self._is_memoized('_odds_positive_spatial'):
            self._odds_positive_spatial = self.prob_positive_spatial / (
                1 - self.prob_positive_spatial
            )
        return self._odds_positive_spatial

    @property
    def zscore_spatial(self) -> np.ndarray:
        if not self._is_memoized('_zscore_spatial'):
            self._zscore_spatial = np.divide(
                self.mean_spatial, self.std_spatial / np.sqrt(self._ds.sizes['time'])
            )

        return self._zscore_spatial

    def get_spatial_metric(self, name: str):
        if isinstance(name, str):
            if name == 'ns_con_var_spatial':
                return self.ns_con_var_spatial
            if name == 'ew_con_var_spatial':
                return self.ew_con_var_spatial
            if name == 'mean_spatial':
                return self.mean_spatial
            if name == 'std_spatial':
                return self.mean_spatial
            if name == 'prob_positive_spatial':
                return self.prob_positive_spatial
            if name == 'prob_negative_spatial':
                return self.prob_negative_spatial
            if name == 'odds_positive_spatial':
                return self.odds_positive_spatial
            if name == 'zscore_spatial':
                return self.zscore_spatial
            raise ValueError(f'there is no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class TimeSeriesMetrics(DatasetMetrics):
    def __init__(self, ds: xr.DataArray):
        DatasetMetrics.__init__(self, ds)
        self._ns_con_var_time = None
        self._ew_con_var_time = None
        self._mean_time = None
        self._std_time = None
        self._prob_positive_time = None
        self._odds_positive_time = None
        self._prob_negative_time = None
        self._zscore_time = None
