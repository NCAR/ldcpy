import numpy as np
import xarray as xr


class Metrics(object):
    def __init__(self, orig: xr.DataArray, compressed: xr.DataArray) -> None:
        if isinstance(orig, xr.DataArray) and isinstance(compressed, xr.DataArray):
            if orig.shape != compressed.shape:
                raise ValueError('both observed and modelled must have the same shape')
            # Datasets
            self._orig = orig
            self._compressed = compressed

            # Variables
            self._ns_con_var_orig = None
            self._ns_con_var_compressed = None
            self._ew_con_var_orig = None
            self._ew_con_var_compressed = None
            self._is_positive_orig = None
            self._is_positive_compressed = None
            self._is_negative_orig = None
            self._is_negative_compressed = None

        else:
            raise TypeError(
                f'both measured and observed must be of type numpy.ndarray. Type(observed): {str(type(orig))}, type(modelled): {str(type(compressed))}'
            )

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    @property
    def ns_con_var_orig(self) -> np.ndarray:
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var_orig = self._con_var('ns', self._orig)

        return self._ns_con_var_orig

    @property
    def ew_con_var_orig(self) -> np.ndarray:
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var_orig = self._con_var('ew', self._orig)

        return self._ew_con_var_orig

    @property
    def ns_con_var_compressed(self) -> np.ndarray:
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var_compressed = self._con_var('ns', self._compressed)

        return self._ns_con_var_compressed

    @property
    def ew_con_var_compressed(self) -> np.ndarray:
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var_compressed = self._con_var('ew', self._compressed)

        return self._ew_con_var_compressed

    @property
    def is_positive_orig(self) -> np.ndarray:
        if not self._is_memoized('_is_positive_orig'):
            self._is_positive_orig = self._orig > 0

        return self._is_positive_orig

    @property
    def is_positive_compressed(self) -> np.ndarray:
        if not self._is_memoized('_is_positive_compressed'):
            self._is_positive_compressed = self._compressed > 0

        return self._is_positive_compressed

    @property
    def is_negative_orig(self) -> np.ndarray:
        if not self._is_memoized('_is_negative_orig'):
            self._is_negative_orig = self._orig < 0

        return self._is_negative_orig

    @property
    def is_negative_compressed(self) -> np.ndarray:
        if not self._is_memoized('_is_negative_compressed'):
            self._is_negative_compressed = self._compressed < 0

        return self._is_negative_compressed

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

    def get_metrics_by_name(self, name: str):
        if isinstance(name, str):
            if name == 'ns_con_var_orig':
                return self.ns_con_var_orig
            if name == 'ns_con_var_compressed':
                return self.ns_con_var_compressed
            if name == 'ew_con_var_orig':
                return self.ew_con_var_orig
            if name == 'ew_con_var_compressed':
                return self.ew_con_var_compressed
            if name == 'is_positive_orig':
                return self._is_positive_orig
            if name == 'is_positive_compressed':
                return self._is_positive_compressed
            raise ValueError(f'there are no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class SpatialMetrics(Metrics):
    def __init__(self, orig: xr.DataArray, compressed: xr.DataArray):
        Metrics.__init__(self, orig, compressed)

        self._ns_con_var_orig_spatial = None
        self._ns_con_var_compressed_spatial = None
        self._ew_con_var_orig_spatial = None
        self._ew_con_var_compressed_spatial = None
        self._mean_orig_spatial = None
        self._mean_compressed_spatial = None
        self._std_orig_spatial = None
        self._std_compressed_spatial = None
        self._prob_positive_orig_spatial = None
        self._prob_positive_compressed_spatial = None
        self._odds_positive_orig_spatial = None
        self._odds_positive_compressed_spatial = None

    @property
    def ns_con_var_orig_spatial(self) -> np.ndarray:
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var_orig_spatial = self._con_var('ns', self._orig).mean(dim='time')

        return self._ns_con_var_orig_spatial

    @property
    def ew_con_var_orig_spatial(self) -> np.ndarray:
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var_orig_spatial = self._con_var('ew', self._orig).mean(dim='time')

        return self._ew_con_var_orig_spatial

    @property
    def ns_con_var_compressed_spatial(self) -> np.ndarray:
        if not self._is_memoized('_ns_con_var'):
            self._ns_con_var_compressed_spatial = self._con_var('ns', self._compressed).mean(
                dim='time'
            )

        return self._ns_con_var_compressed_spatial

    @property
    def ew_con_var_compressed_spatial(self) -> np.ndarray:
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var_compressed_spatial = self._con_var('ew', self._compressed).mean(
                dim='time'
            )

        return self._ew_con_var_compressed_spatial

    @property
    def mean_orig_spatial(self) -> np.ndarray:
        if not self._is_memoized('_mean_orig_spatial'):
            self._mean_orig_spatial = self._orig.mean(dim='time')

        return self._mean_orig_spatial

    @property
    def mean_compressed_spatial(self) -> np.ndarray:
        if not self._is_memoized('_mean_compressed_spatial'):
            self._mean_compressed_spatial = self._compressed.mean(dim='time')

        return self._mean_compressed_spatial

    @property
    def std_orig_spatial(self) -> np.ndarray:
        if not self._is_memoized('_std_orig_spatial'):
            self._std_orig_spatial = self._orig.std(dim='time', ddof=1)

        return self._std_orig_spatial

    @property
    def std_compressed_spatial(self) -> np.ndarray:
        if not self._is_memoized('_std_compressed_spatial'):
            self._std_compressed_spatial = self._compressed.std(dim='time', ddof=1)

        return self._std_compressed_spatial

    @property
    def prob_positive_orig_spatial(self) -> np.ndarray:
        if not self._is_memoized('_prob_positive_orig_spatial'):
            self._prob_positive_orig_spatial = (self.is_positive_orig.sum(dim='time')) / (
                self.is_positive_orig.sizes['time']
            )
        return self._prob_positive_orig_spatial

    @property
    def prob_positive_compressed_spatial(self) -> np.ndarray:
        if not self._is_memoized('_prob_positive_compressed_spatial'):
            self._prob_positive_compressed_spatial = (
                self.is_positive_compressed.sum(dim='time')
            ) / (self.is_positive_compressed.sizes['time'])
        return self._prob_positive_compressed_spatial

    @property
    def prob_negative_orig_spatial(self) -> np.ndarray:
        if not self._is_memoized('_prob_negative_orig_spatial'):
            self._prob_negative_orig_spatial = (
                self.is_negative_orig.sum(dim='time') / self.is_negative_orig.sizes['time']
            )
        return self._prob_negative_orig_spatial

    @property
    def prob_negative_compressed_spatial(self) -> np.ndarray:
        if not self._is_memoized('_prob_negative_compressed_spatial'):
            self._prob_negative_compressed_spatial = (
                self.is_negative_compressed.sum(dim='time')
                / self.is_negative_compressed.sizes['time']
            )
        return self._prob_negative_compressed_spatial

    @property
    def odds_positive_orig_spatial(self) -> np.ndarray:
        if not self._is_memoized('_odds_positive_orig_spatial'):
            self._odds_positive_orig_spatial = self.prob_positive_orig_spatial / (
                1 - self.prob_positive_orig_spatial
            )
        return self._odds_positive_orig_spatial

    @property
    def odds_positive_compressed_spatial(self) -> np.ndarray:
        if not self._is_memoized('_odds_positive_compressed_spatial'):
            self._odds_positive_compressed_spatial = self.prob_positive_compressed_spatial / (
                1 - self.prob_positive_compressed_spatial
            )
        return self._odds_positive_compressed_spatial

    def get_spatial_metrics_by_name(self, name: str):
        if isinstance(name, str):
            if name == 'ns_con_var_orig_spatial':
                return self.ns_con_var_orig_spatial
            if name == 'ns_con_var_compressed_spatial':
                return self.ns_con_var_compressed_spatial
            if name == 'ew_con_var_orig_spatial':
                return self.ew_con_var_orig_spatial
            if name == 'ew_con_var_compressed_spatial':
                return self.ew_con_var_compressed_spatial
            if name == 'mean_orig_spatial':
                return self.mean_orig_spatial
            if name == 'mean_compressed_spatial':
                return self.mean_compressed_spatial
            if name == 'std_orig_spatial':
                return self.mean_orig_spatial
            if name == 'std_compressed_spatial':
                return self.mean_compressed_spatial
            if name == 'prob_positive_orig_spatial':
                return self.prob_positive_orig_spatial
            if name == 'prob_positive_compressed_spatial':
                return self.prob_positive_compressed_spatial
            if name == 'prob_negative_orig_spatial':
                return self.prob_negative_orig_spatial
            if name == 'prob_negative_compressed_spatial':
                return self.prob_negative_compressed_spatial
            if name == 'odds_positive_compressed_spatial':
                return self.odds_positive_orig_spatial
            if name == 'odds_positive_compressed_spatial':
                return self.prob_positive_compressed_spatial
            raise ValueError(f'there is no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class TimeSeriesMetrics(Metrics):
    def __init__(self, orig: xr.DataArray, compressed: xr.DataArray):
        Metrics.__init__(orig, compressed)
