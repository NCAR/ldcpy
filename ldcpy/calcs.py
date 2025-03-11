import copy
import ctypes
import gzip
import struct
import tempfile
import time
from math import exp, pi, sqrt
from typing import Optional

import cf_xarray as cf
import dask
import matplotlib as mpl
import numpy as np
import pandas as pd
import skimage.io
import skimage.metrics
import statsmodels.api as sm
import xarray as xr
from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans
from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib import pyplot as plt
from scipy import stats as ss
from scipy.ndimage import gaussian_filter
from scipy.signal import stft
from skimage.metrics import structural_similarity as ssim
from skimage.util import crop
from xrft import dft

# from .collect_datasets import collect_datasets

xr.set_options(keep_attrs=True)


class Datasetcalcs:
    """
    This class contains calcs for each point of a dataset after aggregating across one or more dimensions, and a method to access these calcs. Expects a DataArray.
    """

    def __init__(
        self,
        ds: xr.DataArray,
        data_type: str,
        aggregate_dims: list,
        time_dim_name: str = None,
        lat_dim_name: str = None,
        lon_dim_name: str = None,
        vert_dim_name: str = None,
        lat_coord_name: str = None,
        lon_coord_name: str = None,
        q: float = 0.5,
        weighted=True,
    ):
        self._ds = ds if (ds.dtype == np.float64) else ds.astype(np.float64)
        # For some reason, casting to float64 removes all attrs from the dataset
        self._ds.attrs = ds.attrs

        if data_type == 'wrf':
            weighted = False

        if weighted:
            if 'cell_measures' not in self._ds.attrs:
                self._ds.attrs['cell_measures'] = 'area: cell_area'

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

        # WRF ALSO HAS XLAT_U and XLONG_U, XLAT_v and XLONG_V
        dd = ds.cf[ds.cf.coordinates['latitude'][0]].dims

        ll = len(dd)
        if data_type == 'cam-fv':  # ll == 1:
            if lat_dim_name is None:
                lat_dim_name = dd[0]
            if lon_dim_name is None:
                lon_dim_name = ds.cf['longitude'].dims[0]
        elif data_type == 'pop':  # ll == 2:
            if lat_dim_name is None:
                lat_dim_name = dd[0]
            if lon_dim_name is None:
                lon_dim_name = dd[1]
        elif data_type == 'wrf':
            if lat_dim_name is None:
                lat_dim_name = dd[0]
            if lon_dim_name is None:
                lon_dim_name = dd[1]
        else:
            print('Warning: unknown data_type: ', data_type)

        self._latlon_dims = ll
        self._lat_dim_name = lat_dim_name
        self._lon_dim_name = lon_dim_name

        # vertical dimension?
        if vert_dim_name is None:
            if data_type == 'wrf':
                vert = 'z' in ds.dims
                if vert:
                    vert_dim_name = 'z'
            else:
                vert = 'vertical' in ds.cf
                if vert:
                    vert_dim_name = ds.cf['vertical'].name
        self._vert_dim_name = vert_dim_name

        # time dimension
        if time_dim_name is None:
            if 'time' in ds.cf.coordinates.keys():
                time_dim_name = ds.cf.coordinates['time'][0]
            else:
                time_dim_name = None

        self._time_dim_name = time_dim_name

        self._quantile = q
        self._agg_dims = aggregate_dims
        self._frame_size = 1
        self._data_type = data_type

        # array calcs
        self._weighted = weighted
        self._ns_con_var = None
        self._ew_con_var = None
        self._mean = None
        self._mean_abs = None
        self._std = None
        self._ddof = 1
        self._num_positive = None
        self._num_negative = None
        self._num_zero = None
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
        self._min_abs_nonzero = None
        self._min_val_nonzero = None
        self._d_range = None
        self._min_val = None
        self._max_val = None
        self._grouping = None
        self._annual_harmonic_relative_ratio = None
        self._lon_autocorr = None
        self._lat_autocorr = None
        self._lev_autocorr = None
        self._entropy = None
        self._w_e_first_differences = None
        self._w_e_first_differences_max = None
        self._n_s_first_differences = None
        self._n_s_first_differences_max = None
        self._w_e_derivative = None
        self._fft2 = None
        self._percent_unique = None
        self._most_repeated = None
        self._most_repeated_pct = None
        self._cdf = None
        self._dtype = ds.dtype
        self._stft = None
        self._fftratio = None
        self._fftmax = None
        self._vfftratio = None
        self._vfftmax = None
        self._magnitude_range = None
        self._magnitude_diff_ew = None
        self._magnitude_diff_ns = None
        self._real_information = None
        self._real_information_cutoff = None

        # single value calcs
        self._zscore_cutoff = None
        self._zscore_percent_significant = None

        adims = self._agg_dims
        self._not_agg_dims = []
        if adims is None:
            adims = self._ds.dims
        else:
            for d in self._ds.dims:
                if d not in adims:
                    self._not_agg_dims.append(d)

        # for probability functions, what is the size
        if aggregate_dims is not None:
            for dim in aggregate_dims:
                self._frame_size *= int(self._ds.sizes[dim])
        else:
            dp = np.count_nonzero(~np.isnan(self._ds))
            self._frame_size = dp

    def _is_memoized(self, calc_name: str) -> bool:
        return hasattr(self, calc_name) and (self.__getattribute__(calc_name) is not None)

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
    def dtype(self) -> xr.DataArray:
        return self._dtype

    @property
    def pooled_variance(self) -> xr.DataArray:
        """
        The overall variance of the dataset
        """
        if not self._is_memoized('_pooled_variance_mean'):
            self._pooled_variance = self._ds.var(self._agg_dims)
            self._pooled_variance.attrs['cell_measures'] = self._ds.attrs['cell_measures']
            if self._weighted:
                self._pooled_variance_mean = self._pooled_variance.cf.weighted('area').mean()
            else:
                self._pooled_variance_mean = self._pooled_variance.mean()
            self._pooled_variance_mean.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._pooled_variance_mean.attrs['units'] = f'{self._ds.units}$^2$'

        return self._pooled_variance_mean

    @property
    def w_e_first_differences(self) -> xr.DataArray:
        """
        First differences along the west-east direction
        """
        if not self._is_memoized('_w_e_first_differences'):
            # self._first_differences = self._ds.diff('lat').mean(self._agg_dims)
            self._w_e_first_differences = self._ds.roll(
                {self._lat_dim_name: -1}, roll_coords=False
            ) - self._ds.roll({self._lat_dim_name: 1}, roll_coords=False)
        self._w_e_first_differences.attrs = self._ds.attrs

        return self._w_e_first_differences.mean(self._agg_dims)

    @property
    def w_e_first_differences_max(self) -> xr.DataArray:
        """
        First differences along the west-east direction
        """
        if not self._is_memoized('_w_e_first_differences'):
            # self._first_differences = self._ds.diff('lat').mean(self._agg_dims)
            self._w_e_first_differences = self._ds.roll(
                {self._lat_dim_name: -1}, roll_coords=False
            ) - self._ds.roll({self._lat_dim_name: 1}, roll_coords=False)
        self._w_e_first_differences.attrs = self._ds.attrs

        return self._w_e_first_differences.max(self._agg_dims)

    @property
    def n_s_first_differences(self) -> xr.DataArray:
        """
        First differences along the north-south direction
        """
        if not self._is_memoized('_n_s_first_differences'):
            self._n_s_first_differences = self._ds.diff(self._lon_dim_name).mean(self._agg_dims)
            # self._first_differences = self._ds.roll({"lon": -1}, roll_coords=False) - self._ds.roll({"lat": 1},                                                                                        roll_coords=False)
        self._n_s_first_differences.attrs = self._ds.attrs
        return self._n_s_first_differences

    @property
    def n_s_first_differences_max(self) -> xr.DataArray:
        """
        First differences along the n-s direction
        """
        if not self._is_memoized('_n_s_first_differences'):
            self._n_s_first_differences = self._ds.diff(self._lon_dim_name).max(self._agg_dims)
            # self._first_differences = self._ds.roll({"lon": -1}, roll_coords=False) - self._ds.roll({"lat": 1},                                                                                        roll_coords=False)
        self._n_s_first_differences.attrs = self._ds.attrs
        return self._n_s_first_differences

    @property
    def percent_unique(self) -> xr.DataArray:
        """
        Percentage of unique values in the dataset
        """
        if not self._is_memoized('_percent_unique'):
            count_unique = len(pd.unique(self._ds.values.flatten()))
            count_all = len(self._ds.values.flatten())
            self._percent_unique = count_unique / count_all

        return self._percent_unique

    @property
    def most_repeated(self) -> xr.DataArray:
        """
        Most repeated value in dataset
        """
        if not self._is_memoized('_most_repeated'):
            self._most_repeated = ss.mode(self._ds.values.flatten())[0]
        return self._most_repeated

    @property
    def most_repeated_percent(self) -> xr.DataArray:
        """
        Most repeated value in dataset
        """
        if not self._is_memoized('_most_repeated_pct'):
            self._most_repeated_pct = ss.mode(self._ds.values.flatten())[1][0] / len(
                self._ds.values.flatten()
            )
        return self._most_repeated_pct

    # def log_max(ds):
    #    a_d = abs(ds)
    #    return np.log10(a_d, where=a_d.data > 0 ).max(skipna=True)
    # return np.log10(abs(ds)).where(np.log10(abs(ds)) != -np.inf).max(skipna=True)

    @property
    def magnitude_range(self) -> xr.DataArray:
        """
        The range of the dataset
        """
        if not self._is_memoized('_magnitude_range'):
            mag_min = self.min_abs_nonzero
            mag_max = self.max_abs
            mag_range = np.log10(mag_max) - np.log10(mag_min)
            # print('RANGE = ', mag_range)
            self._magnitude_range = mag_range

            # Get the range in exponent space
            # def max_agg(ds):
            #    return ds.max(skipna=True)

            # def min_agg(ds):
            #    return ds.min(skipna=True)

            # avoid divde by zero warning
            # a_d = abs(self._ds.copy())
            # print('a_d = ', a_d.data)
            # log_ds = np.log10(a_d, where=a_d.data > 0)
            # print('log_ds = ', log_ds.data)
            # print('data: min abs nonzero = ', self.min_abs_nonzero)
            # print('data: max value = ', self.max_val)
            # if len(self._not_agg_dims) == 0:
            #    my_max = max_agg(log_ds)
            #    my_min = min_agg(log_ds)
            # else:
            #    stack = log_ds.stack(multi_index=tuple(self._not_agg_dims))
            #    my_max = stack.groupby('multi_index').map(max_agg)
            #    my_min = stack.groupby('multi_index').map(min_agg)
            # print('max exp= ', my_max)
            # print('min exp = ', my_min)
            # if (
            #    np.isinf(my_max).any()
            #    or np.isinf(my_min).any()
            #    or np.isnan(my_max).any()
            #    or np.isnan(my_min).any()
            # ):
            #    self._magnitude_range = -1
            #    return self._magnitude_range
            # else:

            # self._magnitude_range = my_max - my_min

        return self._magnitude_range

    @property
    def magnitude_diff_ew(self) -> xr.DataArray:
        """
        Maximum magnitude differences along ew direction
        """
        if not self._is_memoized('_magnitude_diff_ew'):
            # self._first_differences = self._ds.diff('lat').mean(self._agg_dims)
            here = np.log10(abs(self._ds.roll({'lon': 0}, roll_coords=False)))
            there = np.log10(abs(self._ds.roll({'lon': 1}, roll_coords=False)))
            if (
                np.isinf(here).any()
                or np.isinf(there).any()
                or np.isnan(here).any()
                or np.isnan(there).any()
            ):
                self._magnitude_diff_ew_int = -1
                return xr.DataArray(self._magnitude_diff_ew_int)
            else:
                self._magnitude_diff_ew_int = here - there
        self._magnitude_diff_ew = xr.DataArray(self._magnitude_diff_ew_int)
        self._magnitude_diff_ew.attrs = self._ds.attrs
        return self._magnitude_diff_ew.max(self._agg_dims, skipna=True)

    @property
    def magnitude_diff_ns(self) -> xr.DataArray:
        """
        Maximum magnitude irst difference along the n-s direction
        """
        if not self._is_memoized('_magnitude_diff_ns'):
            my_max = np.log10(abs(self._ds.diff('lat').max(self._agg_dims)))
            my_min = np.log10(abs(self._ds.diff('lat').min(self._agg_dims)))
            print('max = ', my_max)
            print('min = ', my_min)
            if (
                np.isinf(my_max).any()
                or np.isinf(my_min).any()
                or np.isnan(my_max).any()
                or np.isnan(my_min).any()
            ):
                self._magnitude_diff_ns_int = -1
                return xr.DataArray(self._magnitude_diff_ns_int)
            else:
                self._magnitude_diff_ns_int = my_max - my_min
            # self._first_differences = self._ds.roll({"lon": -1}, roll_coords=False) - self._ds.roll({"lat": 1},                                                                                        roll_coords=False)
        self._magnitude_diff_ns = xr.DataArray(self._magnitude_diff_ns_int)
        self._magnitude_diff_ns.attrs = self._ds.attrs
        return self._magnitude_diff_ns.max(self._agg_dims, skipna=True)

    @property
    def range(self) -> xr.DataArray:
        """
        The range of the dataset
        """
        return self.max_val - self.min_val

    @property
    def w_e_derivative(self) -> xr.DataArray:
        """
        Derivative of dataset from west-east
        """

        if not self._is_memoized('_derivative'):
            self._derivative = self._ds.differentiate('lon').mean(self._agg_dims)
        self._derivative.attrs = self._ds.attrs

        return self._derivative

    @property
    def ns_con_var(self) -> xr.DataArray:
        """
        The North-South Contrast Variance averaged along the aggregate dimensions
        """
        if not self._is_memoized('_ns_con_var_mean'):
            self._ns_con_var = self._con_var('ns', self._ds)
            if self._weighted:
                self._ns_con_var.attrs['cell_measures'] = self._ds.attrs['cell_measures']
                adims = self._agg_dims
                if adims is None:
                    self._ns_con_var_mean = self._ns_con_var.cf.weighted('area').mean(skipna=True)
                else:
                    self._ns_con_var_mean = self._ns_con_var.cf.weighted('area').mean(
                        dim=adims, skipna=True
                    )
            else:
                self._ns_con_var_mean = self._ns_con_var.mean(self._agg_dims)

            self._ns_con_var_mean.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._ns_con_var_mean.attrs['units'] = f'{self._ds.units}$^2$'

        return self._ns_con_var_mean

    @property
    def ew_con_var(self) -> xr.DataArray:
        """
        The East-West Contrast Variance averaged along the aggregate dimensions
        """
        if not self._is_memoized('_ew_con_var'):
            self._ew_con_var = self._con_var('ew', self._ds)

            if self._weighted:
                self._ew_con_var.attrs['cell_measures'] = self._ds.attrs['cell_measures']
                adims = self._agg_dims
                if adims is None:
                    self._ew_con_var_mean = self._ew_con_var.cf.weighted('area').mean(skipna=True)
                else:
                    self._ew_con_var_mean = self._ew_con_var.cf.weighted('area').mean(
                        dim=adims, skipna=True
                    )
            else:
                self._ew_con_var_mean = self._ew_con_var.mean(self._agg_dims)
            self._ew_con_var_mean.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._ew_con_var_mean.attrs['units'] = f'{self._ds.units}$^2$'

        return self._ew_con_var_mean

    @property
    def lat_autocorr(self) -> xr.DataArray:
        """
        Autocorrelation: the correlation of a variable with itself shifted in the latitude dimension
        """
        if not self._is_memoized('_lat_autocorr'):
            dims = self._ds.dims
            lat_dim_name = self._lat_dim_name
            if lat_dim_name not in dims:
                print('error: latitude not found for autocorrelation')
                self._lat_autocorr = 0
            else:
                idx = dims.index(lat_dim_name)
                sz = self._ds.sizes[lat_dim_name]
                y_lat = np.roll(self._ds, -1, axis=idx)
                x_lat = self._ds
                # this one (longitude) should not wrap
                yd_lat = np.delete(y_lat, sz - 1, axis=idx)
                xd_lat = np.delete(x_lat, sz - 1, axis=idx)
                y_lat_r = np.ravel(yd_lat)
                x_lat_r = np.ravel(xd_lat)
                aa = np.corrcoef(x_lat_r, y_lat_r)[0, 1]
                self._lat_autocorr = aa

        return self._lat_autocorr

    @property
    def lon_autocorr(self) -> xr.DataArray:
        """
        Autocorrelation: the correlation of a variable with itself shifted in the longitude dimension
        """
        if not self._is_memoized('_lon_autocorr'):
            dims = self._ds.dims
            lon_dim_name = self._lon_dim_name
            if lon_dim_name not in dims:
                print('error: longitude not found for autocorrelation')
                self._lon_autocorr = 0
            else:
                idx = dims.index(lon_dim_name)
                y_lon = np.roll(self._ds, -1, axis=idx)
                x_lon = self._ds
                y_lon_r = np.ravel(y_lon)
                x_lon_r = np.ravel(x_lon)
                aa = np.corrcoef(x_lon_r, y_lon_r)[0, 1]
                self._lon_autocorr = aa

        return self._lon_autocorr

    @property
    def lev_autocorr(self) -> xr.DataArray:
        """
        Autocorrelation: the correlation of a variable with itself shifted in the vertical dimension
        """
        if not self._is_memoized('_lev_autocorr'):
            dims = self._ds.dims
            vert_dim_name = self._vert_dim_name
            if vert_dim_name not in dims:
                print('error: vertical (lev) not found for autocorrelation')
                self._lon_autocorr = 0
            else:
                idx = dims.index(vert_dim_name)
                sz = self._ds.sizes[vert_dim_name]
                # print(idx)
                y = np.roll(self._ds, -1, axis=idx)
                x = self._ds
                # this one (vert level) should not wrap
                yd = np.delete(y, sz - 1, axis=idx)
                xd = np.delete(x, sz - 1, axis=idx)
                yd_r = np.ravel(yd)
                xd_r = np.ravel(xd)
                aa = np.corrcoef(xd_r, yd_r)[0, 1]

                self._lev_autocorr = aa

        return self._lev_autocorr

    @property
    def entropy(self) -> xr.DataArray:
        """
        An estimate for the entropy of the data (using gzip)
        # lower is better (1.0 means random - no compression possible)
        """
        if not self._is_memoized('_entropy'):
            es = []

            a1 = self._ds.data
            if dask.is_dask_collection(a1):
                a1 = a1.compute()

            if len(self._not_agg_dims) == 0:
                # don't stack at all, just make a single multi_index group
                cc = gzip.compress(a1, mtime=0)
                dd = gzip.decompress(cc)
                cl = len(cc)
                dl = len(dd)
                if dl > 0:
                    e = cl / dl
                else:
                    e = 0.0
                es.append(e)
            else:
                stack = a1.stack(multi_index=tuple(self._not_agg_dims))
                for d, slice in stack.groupby('multi_index'):
                    cc = gzip.compress(slice.data)
                    dd = gzip.decompress(cc)
                    cl = len(cc)
                    dl = len(dd)
                    if dl > 0:
                        e = cl / dl
                    else:
                        e = 0.0
                    es.append(e)

            if len(self._not_agg_dims) == 0:
                self._entropy = xr.DataArray(es)
            else:
                self._entropy = xr.DataArray(es, dims=self._not_agg_dims)
            self._entropy.attrs = self._ds.attrs
            self._entropy.attrs['units'] = ''
        return self._entropy

    @property
    def mean(self) -> xr.DataArray:
        """
        The mean along the aggregate dimensions
        """
        if not self._is_memoized('_mean'):
            if self._weighted:
                adims = self._agg_dims
                if adims is None:
                    self._mean = self._ds.cf.weighted('area').mean(skipna=True)
                else:
                    self._mean = self._ds.cf.weighted('area').mean(dim=adims, skipna=True)
            else:
                self._mean = self._ds.mean(self._agg_dims, skipna=True)
            self._mean.attrs = self._ds.attrs

        return self._mean

    @property
    def mean_abs(self) -> xr.DataArray:
        """
        The mean of the absolute errors along the aggregate dimensions
        """
        if not self._is_memoized('_mean_abs'):
            if self._weighted:
                adims = self._agg_dims
                if adims is None:
                    self._mean_abs = abs(self._ds).cf.weighted('area').mean(skipna=True)
                else:
                    self._mean_abs = abs(self._ds).cf.weighted('area').mean(dim=adims, skipna=True)
            else:
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
            self.mean_squared.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self.mean_squared.attrs['units'] = f'{self._ds.units}$^2$'

        return self._mean_squared

    @property
    def root_mean_squared(self) -> xr.DataArray:
        """
        The absolute value of the mean along the aggregate dimensions
        """
        if not self._is_memoized('_root_mean_squared_mean'):
            self._squared = np.square(self._ds)

            if self._weighted:
                adims = self._agg_dims
                if adims is None:
                    self._root_mean_squared = np.sqrt(
                        self._squared.cf.weighted('area').mean(skipna=True)
                    )
                else:
                    self._root_mean_squared = np.sqrt(
                        self._squared.cf.weighted('area').mean(dim=adims, skipna=True)
                    )
            else:
                self._root_mean_squared = np.sqrt(self._squared.mean(self._agg_dims, skipna=True))
            self._root_mean_squared.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._root_mean_squared.attrs['units'] = f'{self._ds.units}'

        return self._root_mean_squared

    @property
    def sum(self) -> xr.DataArray:
        if not self._is_memoized('_sum'):
            if self._weighted:
                self._sum = self._ds.sum(dim=self._agg_dims, skipna=True)
            else:
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
            if self._weighted:
                adims = self._agg_dims
                if self._ddof == 0:
                    # biased std
                    if adims is None:
                        self._std = np.sqrt(
                            ((self._ds - self.mean) ** 2).cf.weighted('area').mean()
                        )
                    else:
                        self._std = np.sqrt(
                            ((self._ds - self.mean) ** 2)
                            .cf.weighted('area')
                            .mean(dim=self._agg_dims)
                        )
                elif adims is not None:
                    if 'lat' in adims:
                        # assume unbiased std (reliability weighted)
                        V1 = self._ds.coords['cell_area'].sum(dim='lat')[0]
                        V2 = np.square(self._ds.coords['cell_area']).sum(dim='lat')[0]
                        _biased_var = (
                            ((self._ds - self.mean) ** 2).cf.weighted('area').mean(dim=adims)
                        )
                        self._std = np.sqrt(_biased_var * (1 / (1 - V2 / (V1**2))))
                    else:
                        # same as unweighted
                        self._std = self._ds.std(adims, ddof=self._ddof, skipna=True)
                else:
                    # same as unweighted
                    self._std = self._ds.std(adims, ddof=self._ddof, skipna=True)
            else:
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
                if self._weighted:
                    adims = self._agg_dims
                    if adims is None:
                        self._standardized_mean = (
                            self.mean - self._ds.cf.weighted('area').mean(skipna=True)
                        ) / self.std
                    else:
                        self._standardized_mean = (
                            self.mean - self._ds.cf.weighted('area').mean(dim=adims, skipna=True)
                        ) / self.std
                else:
                    self._standardized_mean = (self.mean - self._ds.mean()) / self._ds.std(ddof=1)
            else:
                grouped = self.mean.groupby(self._grouping)
                grouped_mean = grouped.mean()

                self._standardized_mean = (
                    grouped_mean - grouped_mean.mean(skipna=True)
                ) / grouped_mean.std(ddof=1)
            if hasattr(self._ds, 'units'):
                self._standardized_mean.attrs['units'] = ''

        return self._standardized_mean

    @property
    def variance(self) -> xr.DataArray:
        """
        The variance along the aggregate dimensions
        """
        if not self._is_memoized('_variance'):
            if self._weighted:
                self._variance = np.square(self.std)
            else:
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
            denom = self.pooled_variance.copy().fillna(0)
            denom = np.where(denom == 0, 1e-12, denom)
            self._pooled_variance_ratio = self.variance / denom
            # self._pooled_variance_ratio = self.variance / self.pooled_variance
            self._pooled_variance_ratio.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._pooled_variance_ratio.attrs['units'] = ''

        return self._pooled_variance_ratio

    @property
    def num_positive(self) -> xr.DataArray:
        """
        The probability that a point is positive
        """
        if not self._is_memoized('_num_positive'):
            if self._weighted:
                self._num_positive = (self._ds > 0).sum(self._agg_dims)
            else:
                self._num_positive = (self._ds > 0).sum(self._agg_dims)
        return self._num_positive

    @property
    def num_negative(self) -> xr.DataArray:
        """
        The probability that a point is negative
        """
        if not self._is_memoized('_num_negative'):
            if self._weighted:
                self._num_negative = (self._ds < 0).sum(self._agg_dims)
            else:
                self._num_negative = (self._ds < 0).sum(self._agg_dims)
        return self._num_negative

    @property
    def num_zero(self) -> xr.DataArray:
        """
        The probability that a point is zero
        """
        if not self._is_memoized('_num_zero'):
            if self._weighted:
                self._num_zero = (self._ds == 0).sum(self._agg_dims)
            else:
                self._num_zero = (self._ds == 0).sum(self._agg_dims)
        return self._num_zero

    @property
    def prob_positive(self) -> xr.DataArray:
        """
        The probability that a point is positive
        """
        if not self._is_memoized('_prob_positive'):
            self._prob_positive = self.num_positive / self._frame_size
            # print(self._frame_size)
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
            self._prob_negative = self.num_negative / self._frame_size
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
            if self._time_dim_name is not None:
                self._zscore = np.divide(
                    self.mean, self.std / np.sqrt(self._ds.sizes[self._time_dim_name])
                )
                self._zscore.attrs = self._ds.attrs
                if hasattr(self._ds, 'units'):
                    self._zscore.attrs['units'] = ''
            else:
                self._zscore = 0
                print('Warning: Zscore requires a time dimension')
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
    def cdf(self) -> xr.DataArray:
        """
        The empirical CDF of the dataset.
        """
        if not self._is_memoized('_cdf'):
            # ecfd = sm.distributions.ECDF(self._ds)
            x = np.linspace(min(self._ds), max(self._ds))
            self._cdf = sm.distributions.ECDF(self._ds)(x)

        return self._cdf

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
            self._max_abs = abs(self._ds).max(dim=self._agg_dims, skipna=True)
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
    def min_abs_nonzero(self) -> xr.DataArray:
        if not self._is_memoized('_min_abs_nonzero'):
            mx = abs(self._ds).where(self._ds != 0)
            self._min_abs_nonzero = mx.min(dim=self._agg_dims)
            self._min_abs_nonzero.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._min_abs_nonzero.attrs['units'] = f'{self._ds.units}'

        return self._min_abs_nonzero

    @property
    def max_val(self) -> xr.DataArray:
        if not self._is_memoized('_max_val'):
            self._max_val = self._ds.max(dim=self._agg_dims, skipna=True)
            self._max_val.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._max_val.attrs['units'] = f'{self._ds.units}'

        return self._max_val

    @property
    def min_val(self) -> xr.DataArray:
        if not self._is_memoized('_min_val'):
            self._min_val = self._ds.min(dim=self._agg_dims, skipna=True)
            self._min_val.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._min_val.attrs['units'] = f'{self._ds.units}'

        return self._min_val

    @property
    def min_val_nonzero(self) -> xr.DataArray:
        if not self._is_memoized('_min_val_nonzero'):
            mx = self._ds.where(self._ds != 0)
            self._min_val_nonzero = mx.min(dim=self._agg_dims)
            self._min_val_nonzero.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._min_val_nonzero.attrs['units'] = f'{self._ds.units}'

        return self._min_val_nonzero

    @property
    def dyn_range(self) -> xr.DataArray:
        if not self._is_memoized('_range'):
            self._dyn_range = abs((self._ds).max() - (self._ds).min())
            self._dyn_range.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._dyn_range.attrs['units'] = f'{self._ds.units}'

        return self._dyn_range

    def dec_to_binary(self, num):
        b = bin(ctypes.c_uint32.from_buffer(ctypes.c_float(num)).value)
        strip = b.lstrip('0b')
        return strip.zfill(32)

    def dec_to_binary_3(self, num):
        a = time.perf_counter()
        int32bits = np.asarray(num, dtype=np.float32).view(np.int32).item()
        b = time.perf_counter() - a
        print(b)
        return '{:032b}'.format(int32bits)

    #    def get_adj_bit(self, bit_pos):
    #        return [bit_pos[0] - 1, bit_pos[1]]

    def get_dict_list(self, data_array):
        dict_list_H = []
        # To do: should make this work for 64 also
        N_BITS = 32

        # sz = data_array.size
        # Intialize
        for i in range(N_BITS - 1):
            new_dict = {'00': 0, '01': 0, '10': 0, '11': 0}
            dict_list_H.append(new_dict)

        # convert data array values to bits (so len(b) = sz)
        b = np.empty(data_array.size, dtype=object)
        i = 0
        for y in np.nditer(np.asarray(data_array, dtype=np.float32).view(np.int32)):
            b[i] = '{:032b}'.format(y)
            i += 1

        # go through each bit position (i) (except the last)
        for i in range(N_BITS - 1):
            count = 0
            for y in range(1, len(b)):  # go through each data point

                if b[y - 1][i] in ('0', '1') and b[y][i] in ('0', '1'):
                    count += 1
                    dict_list_H[i][b[y - 1][i] + b[y][i]] += 1
                    # dict_list_H[i]['00']
                    # dict_list_H[i]['01']
                    # dict_list_H[i]['10']
                    # dict_list_H[i]['11']

            # turn to fractions (probabilities)
            if count == 0:
                dict_list_H[i]['00'] = 0
                dict_list_H[i]['01'] = 0
                dict_list_H[i]['10'] = 0
                dict_list_H[i]['11'] = 0
            else:
                dict_list_H[i]['00'] /= count
                dict_list_H[i]['01'] /= count
                dict_list_H[i]['10'] /= count
                dict_list_H[i]['11'] /= count

        return dict_list_H

    def get_mutual_info(self, p00, p01, p10, p11):
        p0 = p00 + p10  # current bit is 0
        p1 = p11 + p01  # current bit is 1
        p0_prev = p00 + p01  # prev bit was 0
        p1_prev = p11 + p10  # prev bit was 1

        # From (4) in paper
        mutual_info = 0
        if p00 > 0:
            mutual_info += p00 * np.log2(p00 / p0_prev / p0)
        if p11 > 0:
            mutual_info += p11 * np.log2(p11 / p1_prev / p1)
        if p01 > 0:
            mutual_info += p01 * np.log2(p01 / p0_prev / p1)
        if p10 > 0:
            mutual_info += p10 * np.log2(p10 / p1_prev / p0)

        return mutual_info

    def get_real_info(self):
        # first, flatten the data array by stacking all the dimensions and removing the coordinates
        # like this:        flattened_ds = self._ds.stack(flattened=['lat', 'lon', 'time']).reset_index('flattened')
        # but over any number of dimensions
        flattened_ds = self._ds.stack(flattened=self._ds.dims).reset_index('flattened')

        dict_list_H = self.get_dict_list(flattened_ds)
        # print(dict_list_H)

        mutual_info_array = []
        # here we go thru each bit position
        for bit_pos_dict in dict_list_H:
            p00 = bit_pos_dict['00']
            p01 = bit_pos_dict['01']
            p10 = bit_pos_dict['10']
            p11 = bit_pos_dict['11']

            mutual_info = self.get_mutual_info(p00, p01, p10, p11)

            mutual_info_array.append(mutual_info)

        mutual_info_array = np.array(mutual_info_array)
        # print("MIA =", mutual_info_array)

        return xr.DataArray(mutual_info_array)

    @property
    def real_information(self) -> xr.DataArray:
        if not self._is_memoized('_real_information'):

            self._real_information = self.get_real_info()
            if hasattr(self._ds, 'units'):
                self._real_information.attrs['units'] = f'{self._ds.units}'

        return self._real_information

    @property
    def real_information_cutoff(self) -> xr.DataArray:
        if not self._is_memoized('_real_information_cutoff'):
            # get the first binary digit
            self._real_information_cutoff = 0
            self._captured_information = 0
            # print(self.real_information)
            normalized_information = self.real_information / self.real_information.sum()
            while self._captured_information < 0.99:
                self._captured_information += normalized_information[self._real_information_cutoff]
                self._real_information_cutoff += 1
            return self._real_information_cutoff
            # else:
            #     return 0

    @property
    def lag1(self) -> xr.DataArray:
        """
        The deseasonalized lag-1 autocorrelation value by day of year
        NOTE: This calc returns an array of spatial values as the data set regardless of aggregate dimensions,
        so can only be plotted in a spatial plot.
        """
        if not self._is_memoized('_lag1'):
            if 'dayofyear' in self._ds.attrs.keys():
                key = f'{self._time_dim_name}.dayofyear'
            else:
                key = f'{self._time_dim_name}'
            grouped = self._ds.groupby(key, squeeze=False)
            if self._time_dim_name in self._ds.attrs.keys():
                self._deseas_resid = grouped - grouped.mean(dim=self._time_dim_name)
            else:
                # note: not actually deseasonalized
                self._deseas_resid = grouped.mean(dim=self._time_dim_name) - self._ds.mean()
            time_length = self._deseas_resid.sizes[self._time_dim_name]
            current = self._deseas_resid.head({self._time_dim_name: time_length - 1})
            next_one = self._deseas_resid.shift({self._time_dim_name: -1}).head(
                {self._time_dim_name: time_length - 1}
            )

            num = current.fillna(0).dot(next_one.fillna(0), dim=self._time_dim_name)
            denom = current.fillna(0).dot(current.fillna(0), dim=self._time_dim_name)
            # don't divide by zero :)
            denom = np.where(denom == 0, 1e-12, denom)
            self._lag1 = num / denom

            self._lag1.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._lag1.attrs['units'] = ''

        return self._lag1

    @property
    def lag1_first_difference(self) -> xr.DataArray:
        """
        The deseasonalized lag-1 autocorrelation value of the first difference of the data by day of year
        NOTE: This calc returns an array of spatial values as the data set regardless of aggregate dimensions,
        so can only be plotted in a spatial plot.
        """
        if not self._is_memoized('_lag1_first_difference'):

            if 'dayofyear' in self._ds.attrs.keys():
                key = f'{self._time_dim_name}.dayofyear'
            else:
                key = f'{self._time_dim_name}'

            grouped = self._ds.groupby(key, squeeze=False)
            if self._time_dim_name in self._ds.attrs.keys():
                self._deseas_resid = grouped - grouped.mean(dim=self._time_dim_name)
            else:
                # note: not actually deseasonalized
                self._deseas_resid = grouped.mean(dim=self._time_dim_name) - self._ds.mean()

            time_length = self._deseas_resid.sizes[self._time_dim_name]
            current = self._deseas_resid.head({self._time_dim_name: time_length - 1})
            next_one = self._deseas_resid.shift({self._time_dim_name: -1}).head(
                {self._time_dim_name: time_length - 1}
            )

            first_difference = next_one - current
            first_difference_current = first_difference.head({self._time_dim_name: time_length - 1})
            first_difference_next = first_difference.shift({self._time_dim_name: -1}).head(
                {self._time_dim_name: time_length - 1}
            )

            num = (first_difference_current * first_difference_next).sum(
                dim=[self._time_dim_name], skipna=True
            )

            denom = first_difference_current.dot(first_difference_current, dim=self._time_dim_name)
            # don't divide by zero
            denom = np.where(denom == 0, 1e-12, denom)

            self._lag1_first_difference = num / denom

            self._lag1_first_difference.attrs = self._ds.attrs
            if hasattr(self._ds, 'units'):
                self._lag1_first_difference.attrs['units'] = ''

        return self._lag1_first_difference

    @property
    def fft2(self) -> xr.DataArray:
        if not self._is_memoized('_fft2'):
            self._fft2 = xr.DataArray(np.abs(np.fft.fft2(self._ds)))
            self._fft2.attrs = self._ds.attrs
            # if hasattr(self._ds, 'units'):
            #    self._fft2.attrs['units'] = f'{self._ds.units}'
            self._fft2 = self._fft2.rename(
                {
                    'dim_0': self._time_dim_name,
                    'dim_1': self._lat_dim_name,
                    'dim_2': self._lon_dim_name,
                }
            )
            self._fft2 = self._fft2.assign_coords(
                {
                    self._lat_dim_name: self._ds.coords[self._lat_dim_name],
                    self._lon_dim_name: self._ds.coords[self._lon_dim_name],
                }
            )
        return self._fft2 / np.mean(self._fft2)

    @property
    def fftratio(self) -> xr.DataArray:
        if not self._is_memoized('_fftratio'):
            # get longitude length: self.fft2.shape[1]
            # select second latitude coordinate: self.fft2.isel({"lat":2})
            # select range: self.fft2.where(self.fft2.lat > 1, drop=True)
            # multirange: self.fft2.where(self.fft2.lat > 1, drop=True).where(self.fft2.lon == 3, drop=True)
            top_val = (
                self.fft2.where(
                    self.fft2.latitude > self.fft2.latitude[int((self.fft2.shape[2] - 1) * 3 / 4)],
                    drop=True,
                )
                .max(dim=self._lon_dim_name)
                .mean(dim=self._lat_dim_name)
            )
            bottom_val = (
                self.fft2.where(
                    np.logical_and(
                        self.fft2.latitude > self.fft2.latitude[int((self.fft2.shape[2] - 1) / 2)],
                        self.fft2.latitude
                        <= self.fft2.latitude[int((self.fft2.shape[2] - 1) * 3 / 4)],
                    ),
                    drop=True,
                )
                .max(dim=self._lon_dim_name)
                .mean(dim=self._lat_dim_name)
            )
            # greater than 1 = more higher frequencies than lower frequencies
            self._fftratio = top_val / bottom_val
        return self._fftratio

    @property
    def fftmax(self) -> xr.DataArray:
        if not self._is_memoized('_fftmax'):
            # get longitude length: self.fft2.shape[1]
            # select second latitude coordinate: self.fft2.isel({"lat":2})
            # select range: self.fft2.where(self.fft2.lat > 1, drop=True)
            # multirange: self.fft2.where(self.fft2.lat > 1, drop=True).where(self.fft2.lon == 3, drop=True)
            top_val = (
                self.fft2.where(
                    self.fft2.lat > self.fft2.lat[int((self.fft2.shape[1] - 1) * 15 / 16)],
                    drop=True,
                )
                .max(dim='lon')
                .max()
            )
            bottom_val = (
                self.fft2.where(
                    np.logical_and(
                        self.fft2.lat > self.fft2.lat[int((self.fft2.shape[1] - 1) / 2)],
                        self.fft2.lat <= self.fft2.lat[int((self.fft2.shape[1] - 1) * 9 / 16)],
                    ),
                    drop=True,
                )
                .max(dim='lon')
                .max()
            )
            # greater than 1 = more higher frequencies than lower frequencies
            self._fftmax = top_val / bottom_val
        return self._fftmax

    @property
    def vfftratio(self) -> xr.DataArray:
        if not self._is_memoized('_fftratio'):
            # get longitude length: self.fft2.shape[1]
            # select second latitude coordinate: self.fft2.isel({"lat":2})
            # select range: self.fft2.where(self.fft2.lat > 1, drop=True)
            # multirange: self.fft2.where(self.fft2.lat > 1, drop=True).where(self.fft2.lon == 3, drop=True)
            top_val = (
                self.fft2.where(
                    self.fft2.lon > self.fft2.lon[int((self.fft2.shape[2] - 1) * 3 / 4)], drop=True
                )
                .max(dim='lat')
                .mean()
            )
            bottom_val = (
                self.fft2.where(
                    np.logical_and(
                        self.fft2.lon > self.fft2.lon[int((self.fft2.shape[2] - 1) / 2)],
                        self.fft2.lon <= self.fft2.lon[int((self.fft2.shape[2] - 1) * 3 / 4)],
                    ),
                    drop=True,
                )
                .max(dim='lat')
                .mean()
            )
            # greater than 1 = more higher frequencies than lower frequencies
            self._fftratio = top_val / bottom_val
        return self._fftratio

    @property
    def vfftmax(self) -> xr.DataArray:
        if not self._is_memoized('_fftmax'):
            # get longitude length: self.fft2.shape[1]
            # select second latitude coordinate: self.fft2.isel({"lat":2})
            # select range: self.fft2.where(self.fft2.lat > 1, drop=True)
            # multirange: self.fft2.where(self.fft2.lat > 1, drop=True).where(self.fft2.lon == 3, drop=True)
            top_val = (
                self.fft2.where(
                    self.fft2.lon > self.fft2.lon[int((self.fft2.shape[2] - 1) * 15 / 16)],
                    drop=True,
                )
                .max(dim='lat')
                .max()
            )
            bottom_val = (
                self.fft2.where(
                    np.logical_and(
                        self.fft2.lon > self.fft2.lon[int((self.fft2.shape[2] - 1) / 2)],
                        self.fft2.lon <= self.fft2.lon[int((self.fft2.shape[2] - 1) * 9 / 16)],
                    ),
                    drop=True,
                )
                .max(dim='lat')
                .max()
            )
            # greater than 1 = more higher frequencies than lower frequencies
            self._fftmax = top_val / bottom_val
        return self._fftmax

    @property
    def stft(self) -> xr.DataArray:
        if not self._is_memoized('_stft'):

            f, t, self._stft = np.abs(
                stft(
                    self.mean.isel({'lat': 100}).squeeze(), nperseg=1, nfft=382, detrend='constant'
                )
            )
            self._stft = xr.DataArray(self._stft)
            self._stft.attrs = self._ds.attrs
            # if hasattr(self._ds, 'units'):
            #    self._stft.attrs['units'] = f'{self._ds.units}'
            self._stft = self._stft.rename({'dim_0': 'lat', 'dim_1': 'lon'})
            self._stft = self._stft.assign_coords(
                {'lat': self._ds.coords['lat'], 'lon': self._ds.coords['lon']}
            )
        return self._stft

    @property
    def annual_harmonic_relative_ratio(self) -> xr.DataArray:
        """
        The annual harmonic relative to the average periodogram value
        in a neighborhood of 50 frequencies around the annual frequency
        NOTE: This assumes the values along the "time" dimension are equally spaced.
        NOTE: This calc returns a lat-lon array regardless of aggregate dimensions, so can only be used in a spatial plot.
        """
        if not self._is_memoized('_annual_harmonic_relative_ratio'):
            # drop time coordinate labels or else it will try to parse them as numbers to check spacing and fail

            new_index = [i for i in range(0, self._ds[self._time_dim_name].size)]
            new_ds = self._ds.assign_coords({self._time_dim_name: new_index})

            lon_coord_name = self._lon_coord_name
            lat_coord_name = self._lat_coord_name

            # rechunk to a single time chunk
            new_ds2 = new_ds.chunk({self._time_dim_name: new_ds[self._time_dim_name].size})
            DF = dft(new_ds2, dim=[self._time_dim_name], detrend='constant')
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
            if p.size > 0:
                pval_cutoff = sorted_pvals[p[len(p) - 1]]
                if not (pval_cutoff.size == 0):
                    sig_locs = np.argwhere(pvals <= pval_cutoff)
                    percent_sig = 100 * np.size(sig_locs, 0) / pvals.size
                else:
                    percent_sig = 0
            else:
                percent_sig = 0
            self._zscore_percent_significant = percent_sig

            return self._zscore_percent_significant

    def get_calc_ds(self, calc_name: str, var_name: str) -> xr.Dataset:
        da = self.get_calc(calc_name)
        ds = da.squeeze().to_dataset(name=var_name, promote_attrs=True)
        ds.attrs['data_type'] = da.data_type
        return ds

    def get_calc(self, name: str, q: Optional[int] = 0.5, grouping: Optional[str] = None, ddof=1):
        """
        Gets a calc aggregated across one or more dimensions of the dataset

        Parameters
        ==========
        name : str
            The name of the calc (must be identical to a property name)

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
            if name == 'fft2':
                return self.fft2
            if name == 'stft':
                return self.stft
            if name == 'w_e_first_differences':
                return self.w_e_first_differences
            if name == 'n_s_first_differences':
                return self.n_s_first_differences
            if name == 'w_e_first_differences_max':
                return self.w_e_first_differences_max
            if name == 'n_s_first_differences_max':
                return self.n_s_first_differences_max
            if name == 'w_e_derivative':
                return self.w_e_derivative
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
            if name == 'num_positive':
                return self.num_positive
            if name == 'num_negative':
                return self.num_negative
            if name == 'num_zero':
                return self.num_zero
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
            if name == 'range':
                return self.range
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
            if name == 'min_abs_nonzero':
                return self.min_abs_nonzero
            if name == 'min_abs':
                return self.min_abs
            if name == 'max_val':
                return self.max_val
            if name == 'min_val':
                return self.min_val
            if name == 'min_val_nonzero':
                return self.min_val_nonzero
            if name == 'cdf':
                return self.cdf
            if name == 'fftratio':
                return self.fftratio
            if name == 'fftmax':
                return self.fftmax
            if name == 'vfftratio':
                return self.vfftratio
            if name == 'vfftmax':
                return self.vfftmax
            if name == 'ds':
                return self._ds
            if name == 'magnitude_diff_ew':
                return self.magnitude_diff_ew
            if name == 'magnitude_diff_ns':
                return self.magnitude_diff_ns
            if name == 'real_information':
                return self.real_information
            if name == 'magnitude_range':
                return self.magnitude_range
            raise ValueError(f'there is no calc with the name: {name}.')
        else:
            raise TypeError('name must be a string.')

    def get_single_calc(self, name: str):
        """
        Gets a calc consisting of a single float value

        Parameters
        ==========
        name : str
            the name of the calc (must be identical to a property name)

        Returns
        =======
        out : float
            The calc value
        """
        if isinstance(name, str):
            if name == 'real_information_cutoff':
                return self.real_information_cutoff
            if name == 'zscore_cutoff':
                return self.zscore_cutoff
            if name == 'zscore_percent_significant':
                return self.zscore_percent_significant
            if name == 'lat_autocorr':
                return self.lat_autocorr
            if name == 'lon_autocorr':
                return self.lon_autocorr
            if name == 'lev_autocorr':
                return self.lev_autocorr
            if name == 'entropy':
                return self.entropy
            if name == 'range':
                return self.dyn_range
            if name == 'pooled_variance':
                return self.pooled_variance
            if name == 'annual_harmonic_relative_ratio_pct_sig':
                return self.annual_harmonic_relative_ratio_pct_sig
            if name == 'percent_unique':
                return self.percent_unique
            if name == 'most_repeated':
                return self.most_repeated
            if name == 'most_repeated_percent':
                return self.most_repeated_percent
            if name == 'fftratio':
                return self.fftratio
            if name == 'fftmax':
                return self.fftmax
            if name == 'vfftratio':
                return self.vfftratio
            if name == 'vfftmax':
                return self.vfftmax
            raise ValueError(f'there is no calcs with the name: {name}.')
        else:
            raise TypeError('name must be a string.')


class Diffcalcs:
    """
    This class contains calcs on the overall dataset that require more than one input dataset to compute
    """

    def __init__(
        self,
        ds1: xr.DataArray,
        ds2: xr.DataArray,
        data_type: str,
        aggregate_dims: Optional[list] = None,
        spre_tol: float = 1.0e-4,
        k1: float = 0.01,
        k2: float = 0.03,
        **calcs_kwargs,
    ) -> None:
        if isinstance(ds1, xr.DataArray) and isinstance(ds2, xr.DataArray):
            # Datasets
            self._ds1 = ds1
            self._ds2 = ds2
        else:
            raise TypeError(
                f'ds must be of type xarray.DataArray. Type(s): {str(type(ds1))} {str(type(ds2))}'
            )

        self._calcs1 = Datasetcalcs(self._ds1, data_type, aggregate_dims, **calcs_kwargs)
        self._calcs2 = Datasetcalcs(self._ds2, data_type, aggregate_dims, **calcs_kwargs)
        self._data_type = data_type
        self._aggregate_dims = aggregate_dims
        self._pcc = None
        self._covariance = None
        self._ks_p_value = None
        self._n_rms = None
        self._n_emax = None
        self._spatial_rel_error = None
        self._ssim_value = None  # uses images
        self._ssim_value_fp_orig = (
            None  # "straightforward" version for floating points - not recommended
        )
        self._ssim_value_fp_fast = None  # faster Data SSIM - the default
        self._ssim_value_fp_fast_exp = None  # experimenting
        self._ssim_value_fp_slow = None  # slower non-matrix version of DSSIM - for experimenting
        self._max_spatial_rel_error = None
        self._ssim_mat_fp = None
        self._ssim_mat = None
        self._ssim_mat_fp_orig = None
        self._spre_tol = spre_tol
        self._ssim_levs = None
        self._k1 = k1
        self._k2 = k2
        self._quant = 256

    @property
    def spre_tol(self):
        return self._spre_tol

    @spre_tol.setter
    def spre_tol(self, t):
        self._spre_tol = t

    @property
    def k1(self):
        return self._k1

    @k1.setter
    def k1(self, t):
        self._k1 = t

    @property
    def k2(self):
        return self._k2

    @k2.setter
    def k2(self, t):
        self._k2 = t

    @property
    def quant(self):
        return self._quant

    @quant.setter
    def quant(self, t):
        self._quant = t

    def _is_memoized(self, calc_name: str) -> bool:
        return hasattr(self, calc_name) and (self.__getattribute__(calc_name) is not None)

    # n is the box width (1D)
    # sigma is the radius for the gaussian
    def _oned_gauss(self, n, sigma):
        r = range(-int(n / 2), int(n / 2) + 1)
        return [(1 / (sigma * sqrt(2 * pi))) * exp(-float(x) ** 2 / (2 * sigma**2)) for x in r]

    # this adjusts the boundary area after filtering with astropy
    # to account for the fact that we want to divide only by the weight in
    # the domain at the edges (this is used in fast2)
    def _filter_adjust_edges(self, mydata, kernel, k):
        k_sum = kernel.array.sum()
        X = mydata.shape[0]
        Y = mydata.shape[1]
        # R and L rectangles and T and B rectangles
        for j in range(k):
            kk = kernel.array[:, 0 : k + j + 1].sum()
            scale = k_sum / kk
            # L and R
            mydata[k : X - k, j] = mydata[k : X - k, j] * scale
            mydata[k : X - k, Y - j - 1] = mydata[k : X - k, Y - j - 1] * scale
            # T and B
            mydata[j, k : Y - k] = mydata[j, k : Y - k] * scale
            mydata[X - j - 1, k : Y - k] = mydata[X - j - 1, k : Y - k] * scale

        # corners
        for i in range(k):
            for j in range(k):
                kk = kernel.array[0 : k + i + 1, 0 : k + j + 1].sum()
                scale = k_sum / kk
                # top left
                mydata[i, j] = mydata[i, j] * scale
                # top right
                mydata[i, Y - j - 1] = mydata[i, Y - j - 1] * scale
                # bottom left
                mydata[X - i - 1, j] = mydata[X - i - 1, j] * scale
                # bottom right
                mydata[X - i - 1, Y - j - 1] = mydata[X - i - 1, Y - j - 1] * scale

    def plot_ssim_mat(self, return_mat=False, ssim_type='ssim_fp'):
        if ssim_type == 'orig':
            if self._ssim_value is None:
                self.ssim_value
            mats = self._ssim_mat
        elif ssim_type == 'ssim_fp_orig':
            if self._ssim_value_fp_orig is None:
                self.ssim_value_fp_orig
            mats = self._ssim_mat_fp_orig
        else:
            if self._ssim_value_fp_fast is None:
                self.ssim_value_fp_fast
            mats = self._ssim_mat_fp

        num = len(mats)
        meana = np.zeros(num)
        for i in range(num):
            meana[i] = np.nanmean(mats[i])

        # for 3D which is the min level
        min_meana = meana.min()
        min_lev = meana.argmin()

        # smallest value at that level
        min_lev_val = np.nanmin(mats[min_lev])

        ind = np.unravel_index(np.nanargmin(mats[min_lev], axis=None), mats[min_lev].shape)

        plt.imshow(mats[min_lev], interpolation='none', vmax=1.0, cmap='bone')
        plt.colorbar(orientation='horizontal')
        if num == 1:
            mytitle = f'ssim val = {min_meana:.4f}, min = {min_lev_val:.4f} at {ind}'
        else:
            mytitle = (
                f'ssim val = {min_meana:.4f}, min = {min_lev_val:.4f} at {ind} on lev={min_lev}'
            )

        plt.title(mytitle)
        # plt.show()

        if return_mat:
            return mats

    @property
    def covariance(self) -> xr.DataArray:
        """
        The covariance between the two datasets
        """
        if not self._is_memoized('_covariance'):

            # need to use unweighted means
            c1_mean = self._calcs1.get_calc('ds').mean(dim=self._aggregate_dims, skipna=True)
            c2_mean = self._calcs2.get_calc('ds').mean(dim=self._aggregate_dims, skipna=True)

            self._covariance = (
                (self._calcs2.get_calc('ds') - c2_mean) * (self._calcs1.get_calc('ds') - c1_mean)
            ).mean(dim=self._aggregate_dims)

        return self._covariance

    # @property
    # def ks_p_value(self):
    #     """
    #     The Kolmogorov-Smirnov p-value
    #     """
    #     # Note: ravel() forces a compute for dask, but ks test in scipy can't
    #     # work with uncomputed dask arrays
    #     if not self._is_memoized('_ks_p_value'):
    #         d1_p = (np.ravel(self._ds1)).astype('float64')
    #         d2_p = (np.ravel(self._ds2)).astype('float64')
    #         self._ks_p_value = np.asanyarray(ss.ks_2samp(d2_p, d1_p))
    #     return self._ks_p_value[1]

    @property
    def ks_p_value(self):
        """
        The Kolmogorov-Smirnov p-value, calculated for slices of the data defined by specified dimensions.
        :param dim: Dimensions over which to apply the KS test. Can be a string or a list of strings.
        """
        if not self._is_memoized('_ks_p_value'):
            # Apply the KS test across the specified dimensions
            # This will create a DataArray of p-values

            if self._aggregate_dims is None:
                lat_n = self._ds1.cf['latitude'].name
                lon_n = self._ds2.cf['longitude'].name
                core_d = [[lat_n, lon_n], [lat_n, lon_n]]
            else:
                core_d = [self._aggregate_dims, self._aggregate_dims]
            # print(core_d)
            self._ks_p_value = xr.apply_ufunc(
                lambda x, y: ss.ks_2samp(x.ravel(), y.ravel())[1],
                self._ds1,
                self._ds2,
                input_core_dims=core_d,
                dask='parallelized',
                dask_gufunc_kwargs={'allow_rechunk': True},
                vectorize=True,
            )

        return self._ks_p_value

    @property
    def pearson_correlation_coefficient(self):
        """
        returns the pearson correlation coefficient between the two datasets
        """
        if not self._is_memoized('_pearson_correlation_coefficient'):

            # Calculate standard deviation over the specified dimensions
            c1_std = self._calcs1.get_calc('ds').std(dim=self._aggregate_dims, skipna=True)
            c2_std = self._calcs2.get_calc('ds').std(dim=self._aggregate_dims, skipna=True)

            # Handle cases where standard deviations are zero
            zero_std = (c1_std == 0) | (c2_std == 0)
            self._pcc = xr.where(zero_std, -1.0, self.covariance / c1_std / c2_std)

        return self._pcc

    @property
    def normalized_max_pointwise_error(self):
        """
        The absolute value of the maximum pointwise difference, normalized
        by the range of values for the first set
        """
        if not self._is_memoized('_normalized_max_pointwise_error'):
            tt = abs((self._calcs1.get_calc('ds') - self._calcs2.get_calc('ds')).max())
            self._n_emax = tt / self._calcs1.dyn_range

        return float(self._n_emax)

    @property
    def normalized_root_mean_squared(self):
        """
        The absolute value of the mean along the aggregate dimensions, normalized
        by the range of values for the first set
        """
        if not self._is_memoized('_normalized_root_mean_squared'):
            tt = np.sqrt(
                np.square(self._calcs1.get_calc('ds') - self._calcs2.get_calc('ds')).mean(
                    dim=self._aggregate_dims
                )
            )
            self._n_rms = tt / self._calcs1.dyn_range

        return float(self._n_rms)

    # @property
    # def spatial_rel_error(self):
    #     """
    #     At each grid point, we compute the relative error.  Then we report the percentage of grid point whose
    #     relative error is above the specified tolerance (1e-4 by default).
    #     """
    #
    #     # We don't check for memoization here in case the spre_tol has changed
    #     # since the last time it has been called
    #     sp_tol = self._spre_tol
    #     # unraveling converts the dask array to numpy, but then
    #     # we can assign the 1.0 and avoid zero (couldn't figure another way)
    #     t1 = np.ravel(self._calcs1.get_calc('ds'))
    #     t2 = np.ravel(self._calcs2.get_calc('ds'))
    #
    #     # check for zeros in t1 (if zero then change to 1 - which
    #     # does an absolute error at that point)
    #     z = (np.where(abs(t1) == 0))[0]
    #     if z.size > 0:
    #         t1_denom = np.copy(t1)
    #         t1_denom[z] = 1.0
    #     else:
    #         t1_denom = t1
    #
    #     # we don't want to use nan
    #     # (ocassionally in cam data - often in ocn)
    #     m_t2 = np.ma.masked_invalid(t2).compressed()
    #     m_t1 = np.ma.masked_invalid(t1).compressed()
    #
    #     if m_t2.shape != m_t1.shape:
    #         print('Warning: Spatial error not calculated with differing numbers of Nans')
    #         self._spatial_rel_error = 0
    #         self._max_spatial_rel_error = 0
    #     else:
    #
    #         if z.size > 0:
    #             m_t1_denom = np.ma.masked_invalid(t1_denom).compressed()
    #         else:
    #             m_t1_denom = m_t1
    #
    #         m_tt = m_t1 - m_t2
    #         m_tt = m_tt / m_t1_denom
    #
    #         # find the max spatial error also if None
    #         if self._max_spatial_rel_error is None:
    #             max_spre = np.max(m_tt)
    #             self._max_spatial_rel_error = max_spre
    #
    #         # percentage greater than the tolerance
    #         a = len(m_tt[abs(m_tt) > sp_tol])
    #         sz = m_tt.shape[0]
    #
    #         self._spatial_rel_error = (a / sz) * 100
    #
    #     return float(self._spatial_rel_error)

    @property
    def spatial_rel_error(self):
        """
        At each grid point, we compute the relative error. Then we report the percentage of grid points whose
        relative error is above the specified tolerance (1e-4 by default), optionally calculated over specified dimensions.
        :param dim: Dimensions over which to calculate the spatial relative error. Can be a string or a list of strings.
        """

        sp_tol = self._spre_tol

        t1 = self._calcs1.get_calc('ds')
        t2 = self._calcs2.get_calc('ds')

        # Replace zeros in t1 with 1.0 (to avoid division by zero)
        t1_denom = xr.where(t1 == 0, 1.0, t1)

        # Compute the relative error
        rel_error = xr.where(t1 != 0, abs(t1 - t2) / abs(t1_denom), abs(t1 - t2))

        # Aggregate the relative error over the specified dimensions
        # Here we calculate the fraction of data points with error above tolerance in each slice
        if self._aggregate_dims is not None:
            # Mask invalid values
            rel_error = rel_error.where(rel_error.notnull(), 0)

            # Calculate the fraction of points above the tolerance for each slice
            error_fraction = (rel_error > sp_tol).mean(dim=self._aggregate_dims) * 100
        else:
            # Global calculation (as in the original function)
            valid_points = rel_error.count()
            error_points = rel_error.where(rel_error > sp_tol).count()
            error_fraction = (error_points / valid_points) * 100

        return error_fraction

    @property
    def max_spatial_rel_error(self):
        """
        We compute the relative error at each grid point and return the maximun.
        """
        # this is also computed as part of the spatial rel error
        if not self._is_memoized('_max_spatial_rel_error'):
            t1 = np.ravel(self._calcs1.get_calc('ds'))
            t2 = np.ravel(self._calcs2.get_calc('ds'))
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

        return float(self._max_spatial_rel_error)

    @property
    def ssim_value(self):
        """
        We compute the SSIM (structural similarity index) on the visualization of the spatial data.
        This creates two plots and uses the standard SSIM.
        """

        k1 = self._k1
        k2 = self._k2

        if True:
            # Prevent showing stuff
            backend_ = mpl.get_backend()
            mpl.use('Agg')

            d1 = self._calcs1.get_calc('ds')
            d2 = self._calcs2.get_calc('ds')

            lat1 = d1[self._calcs1._lat_coord_name]
            lat2 = d2[self._calcs2._lat_coord_name]
            lon1 = d1[self._calcs1._lon_coord_name]
            lon2 = d2[self._calcs2._lon_coord_name]

            # latdim = d1.cf[self._calcs1._lon_coord_name].ndim
            central = 0.0  # might make this a parameter later
            if self._data_type == 'pop':
                central = 300.0
            # make periodic for pop or cam-fv
            if self._data_type == 'pop':
                cy_lon1 = np.hstack((lon1, lon1[:, 0:1]))
                cy_lon2 = np.hstack((lon2, lon2[:, 0:1]))
                cy_lat1 = np.hstack((lat1, lat1[:, 0:1]))
                cy_lat2 = np.hstack((lat2, lat2[:, 0:1]))
                cy1 = add_cyclic_point(d1)
                cy2 = add_cyclic_point(d2)
                no_inf_d1 = np.nan_to_num(cy1, nan=np.nan)
                no_inf_d2 = np.nan_to_num(cy2, nan=np.nan)

            elif self._data_type == 'cam-fv':  # cam-fv
                cy1, cy_lon1 = add_cyclic_point(d1, coord=lon1)
                cy2, cy_lon2 = add_cyclic_point(d2, coord=lon2)
                cy_lat1 = lat1
                cy_lat2 = lat2
                no_inf_d1 = np.nan_to_num(cy1, nan=np.nan)
                no_inf_d2 = np.nan_to_num(cy2, nan=np.nan)

            elif self._data_type == 'wrf':
                no_inf_d1 = np.nan_to_num(d1, nan=np.nan)
                no_inf_d2 = np.nan_to_num(d2, nan=np.nan)

            # is it 3D? must do each level
            if self._calcs1._vert_dim_name is not None:
                vname = self._calcs1._vert_dim_name
                if vname not in self._calcs1.get_calc('ds').sizes:
                    nlevels = 1
                else:
                    nlevels = self._calcs1.get_calc('ds').sizes[vname]
            else:
                nlevels = 1

            color_min = min(
                np.min(d1.where(d1 != -np.inf)).values.min(),
                np.min(d2.where(d2 != -np.inf)).values.min(),
            )
            color_max = max(
                np.max(d1.where(d1 != np.inf)).values.max(),
                np.max(d2.where(d2 != np.inf)).values.max(),
            )

            mymap = copy.copy(plt.cm.get_cmap(self.color))
            mymap.set_under(color='black')
            mymap.set_over(color='white')
            mymap.set_bad(alpha=0)

            ssim_levs = np.zeros(nlevels)
            ssim_mats_array = []

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
                        # this assumes 3D level is first (TO DO: verify)
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
                    # scikit is adding an alpha channel of all zeros - get rid of it or it arificially
                    # inflates the ssim
                    img1 = img1[:, :, :3]
                    img2 = img2[:, :, :3]

                    # s = ssim(img1, img2, multichannel=True)
                    # channel_axis =
                    # the following version closer to matlab version (and orig ssim paper)
                    s, ssim_mat = ssim(
                        img1,
                        img2,
                        #                        multichannel=True,
                        channel_axis=2,
                        gaussian_weights=True,
                        use_sample_covariance=False,
                        full=True,
                        K1=k1,
                        K2=k2,
                    )

                    ssim_levs[this_lev] = s
                    plt.close(fig)
                    ssim_mats_array.append(np.mean(ssim_mat, axis=2))

            return_ssim = ssim_levs.min()

            # Reset backend
            mpl.use(backend_)

            # save full matrix
            self._ssim_mat = ssim_mats_array
            self._ssim_levs = ssim_levs

            self._ssim_value = return_ssim

        return float(self._ssim_value)

    @property
    def ssim_value_fp_slow(self):
        """
        We compute the SSIM (structural similarity index) on the spatial data
        - using the data itself (we do not create an image) - this is the slower
        non-matrix implementation that is good for experiementing (not in practice).

        """

        if not self._is_memoized('_ssim_value_fp_slow'):

            # if this is a 3D variable, we will do each level seperately
            if self._calcs1._vert_dim_name is not None:
                vname = self._calcs1._vert_dim_name
                if vname not in self._calcs1.get_calc('ds').sizes:
                    nlevels = 1
                else:
                    nlevels = self._calcs1.get_calc('ds').sizes[vname]
            else:
                nlevels = 1

            ssim_levs = np.zeros(nlevels)
            ssim_mats_array = []

            for this_lev in range(nlevels):
                if nlevels == 1:
                    a1 = self._calcs1.get_calc('ds').data
                    a2 = self._calcs2.get_calc('ds').data
                else:
                    a1 = self._calcs1.get_calc('ds').isel({vname: this_lev}).data
                    a2 = self._calcs2.get_calc('ds').isel({vname: this_lev}).data

                if dask.is_dask_collection(a1):
                    a1 = a1.compute()
                if dask.is_dask_collection(a2):
                    a2 = a2.compute()

                # re-scale  to [0,1] - if not constant
                smin = min(np.nanmin(a1), np.nanmin(a2))
                smax = max(np.nanmax(a1), np.nanmax(a2))
                r = smax - smin
                if r == 0.0:  # scale by smax if field is a constant (and smax != 0)
                    if smax == 0.0:
                        sc_a1 = a1
                        sc_a2 = a2
                    else:
                        sc_a1 = a1 / smax
                        sc_a2 = a2 / smax
                else:
                    sc_a1 = (a1 - smin) / r
                    sc_a2 = (a2 - smin) / r

                # TEMP - don't quantize
                # now quantize to 256 bins
                sc_a1 = np.round(sc_a1 * 255) / 255
                sc_a2 = np.round(sc_a2 * 255) / 255

                # gaussian filter
                n = 11  # recommended window size
                k = 5
                # extent
                sigma = 1.5

                X = sc_a1.shape[0]
                Y = sc_a1.shape[1]

                g_w = np.array(self._oned_gauss(n, sigma))
                # 2D gauss weights
                gg_w = np.outer(g_w, g_w)

                # init ssim matrix
                ssim_mat = np.zeros_like(sc_a1)

                my_eps = 1.0e-8

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
                        a1_std_sq = (
                            np.average((a1_win[indices1] * a1_win[indices1]), weights=Wt[indices1])
                            - a1_mu * a1_mu
                        )
                        a2_std_sq = (
                            np.average((a2_win[indices2] * a2_win[indices2]), weights=Wt[indices2])
                            - a2_mu * a2_mu
                        )

                        # cov of a1 and a2
                        a1a2_cov = (
                            np.average(
                                (a1_win[indices1] * a2_win[indices2]),
                                weights=Wt[indices1],
                            )
                            - a1_mu * a2_mu
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

                # add cropping
                ssim_mat = crop(ssim_mat, k)

                mean_ssim = np.nanmean(ssim_mat)
                ssim_levs[this_lev] = mean_ssim
                ssim_mats_array.append(ssim_mat)

            return_ssim = ssim_levs.min()
            self._ssim_value_fp_slow = return_ssim
            # save full matrix
            self._ssim_mat_fp_slow = ssim_mats_array
            self._ssim_levs = ssim_levs

        return float(self._ssim_value_fp_slow)

    @property
    def xsize(self):
        return self._xsize

    @xsize.setter
    def xsize(self, xsize):
        self._xsize = xsize

    @property
    def ysize(self):
        return self._ysize

    @ysize.setter
    def ysize(self, ysize):
        self._ysize = ysize

    @property
    def ssim_value_fp_fast(self):
        """
        Faster implementation then ssim_value_fp_slow (this is the default DSSIM option).

        """

        #        from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans

        if not self._is_memoized('_ssim_value_fp_fast'):

            # if this is a 3D variable, we will do each level separately
            if self._calcs1._vert_dim_name is not None:
                vname = self._calcs1._vert_dim_name
                if vname not in self._calcs1.get_calc('ds').sizes:
                    nlevels = 1
                else:
                    nlevels = self._calcs1.get_calc('ds').sizes[vname]
            else:
                nlevels = 1

            ssim_levs = np.zeros(nlevels)
            ssim_mats_array = []
            my_eps = 1.0e-8

            for this_lev in range(nlevels):
                if nlevels == 1:
                    a1 = self._calcs1.get_calc('ds').data
                    a2 = self._calcs2.get_calc('ds').data
                else:
                    a1 = self._calcs1.get_calc('ds').isel({vname: this_lev}).data
                    a2 = self._calcs2.get_calc('ds').isel({vname: this_lev}).data

                if dask.is_dask_collection(a1):
                    a1 = a1.compute()
                if dask.is_dask_collection(a2):
                    a2 = a2.compute()

                # re-scale  to [0,1] - if not constant
                smin = min(np.nanmin(a1), np.nanmin(a2))
                smax = max(np.nanmax(a1), np.nanmax(a2))
                r = smax - smin
                if r == 0.0:  # scale by smax if field is a constant (and smax != 0)
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
                kernel = Gaussian2DKernel(x_stddev=1.5, x_size=self.xsize, y_size=self.ysize)
                k = 5
                filter_args = {'boundary': 'fill', 'preserve_nan': True}

                a1_mu = convolve(sc_a1, kernel, **filter_args)
                a2_mu = convolve(sc_a2, kernel, **filter_args)

                a1a1 = convolve(sc_a1 * sc_a1, kernel, **filter_args)
                a2a2 = convolve(sc_a2 * sc_a2, kernel, **filter_args)

                a1a2 = convolve(sc_a1 * sc_a2, kernel, **filter_args)

                ###########
                var_a1 = a1a1 - a1_mu * a1_mu
                var_a2 = a2a2 - a2_mu * a2_mu
                cov_a1a2 = a1a2 - a1_mu * a2_mu

                # ssim constants
                C1 = my_eps
                C2 = my_eps

                ssim_t1 = 2 * a1_mu * a2_mu + C1
                ssim_t2 = 2 * cov_a1a2 + C2

                ssim_b1 = a1_mu * a1_mu + a2_mu * a2_mu + C1
                ssim_b2 = var_a1 + var_a2 + C2

                ssim_1 = ssim_t1 / ssim_b1
                ssim_2 = ssim_t2 / ssim_b2
                ssim_mat = ssim_1 * ssim_2

                # cropping (the border region)
                ssim_mat = crop(ssim_mat, k)

                mean_ssim = np.nanmean(ssim_mat)
                ssim_levs[this_lev] = mean_ssim
                ssim_mats_array.append(ssim_mat)

            # end of levels calculation
            return_ssim = ssim_levs.min()
            self._ssim_value_fp_fast = return_ssim

            # save ssim on each level
            self._ssim_levs = ssim_levs

            # save full matrix
            self._ssim_mat_fp = ssim_mats_array
        return float(self._ssim_value_fp_fast)

    @property
    def ssim_value_fp_fast_exp(self):
        """
        EXP:

        """

        # from astropy.convolution import Gaussian2DKernel, convolve, interpolate_replace_nans

        if not self._is_memoized('_ssim_value_fp_fast_exp'):

            # if this is a 3D variable, we will do each level separately
            if self._calcs1._vert_dim_name is not None:
                vname = self._calcs1._vert_dim_name
                if vname not in self._calcs1.get_calc('ds').sizes:
                    nlevels = 1
                else:
                    nlevels = self._calcs1.get_calc('ds').sizes[vname]
            else:
                nlevels = 1

            ssim_levs = np.zeros(nlevels)
            ssim_mats_array = []
            my_eps = 1.0e-8

            for this_lev in range(nlevels):
                if nlevels == 1:
                    a1 = self._calcs1.get_calc('ds').data
                    a2 = self._calcs2.get_calc('ds').data
                else:
                    a1 = self._calcs1.get_calc('ds').isel({vname: this_lev}).data
                    a2 = self._calcs2.get_calc('ds').isel({vname: this_lev}).data

                if dask.is_dask_collection(a1):
                    a1 = a1.compute()
                if dask.is_dask_collection(a2):
                    a2 = a2.compute()

                # re-scale  to [0,1] - if not constant
                smin = min(np.nanmin(a1), np.nanmin(a2))
                smax = max(np.nanmax(a1), np.nanmax(a2))
                r = smax - smin
                if r == 0.0:  # scale by smax if field is a constant (and smax != 0)
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
                # sc_a1 = np.round(sc_a1 * 255) / 255
                # sc_a2 = np.round(sc_a2 * 255) / 255

                # gaussian filter
                kernel = Gaussian2DKernel(x_stddev=1.5, x_size=11, y_size=11)
                k = 5
                filter_args = {'boundary': 'fill', 'preserve_nan': True}

                a1_mu = convolve(sc_a1, kernel, **filter_args)
                a2_mu = convolve(sc_a2, kernel, **filter_args)

                a1a1 = convolve(sc_a1 * sc_a1, kernel, **filter_args)
                a2a2 = convolve(sc_a2 * sc_a2, kernel, **filter_args)

                a1a2 = convolve(sc_a1 * sc_a2, kernel, **filter_args)

                ###########
                var_a1 = a1a1 - a1_mu * a1_mu
                var_a2 = a2a2 - a2_mu * a2_mu
                cov_a1a2 = a1a2 - a1_mu * a2_mu

                # ssim constants
                C1 = my_eps
                C2 = my_eps

                ssim_t1 = 2 * a1_mu * a2_mu + C1
                ssim_t2 = 2 * cov_a1a2 + C2

                ssim_b1 = a1_mu * a1_mu + a2_mu * a2_mu + C1
                ssim_b2 = var_a1 + var_a2 + C2

                ssim_1 = ssim_t1 / ssim_b1
                ssim_2 = ssim_t2 / ssim_b2
                ssim_mat = ssim_1 * ssim_2

                # cropping (the border region)
                ssim_mat = crop(ssim_mat, k)

                mean_ssim = np.nanmean(ssim_mat)
                ssim_levs[this_lev] = mean_ssim
                ssim_mats_array.append(ssim_mat)

            # end of levels calculation
            return_ssim = ssim_levs.min()
            self._ssim_value_fp_fast_exp = return_ssim

            # save ssim on each level
            self._ssim_levs = ssim_levs

            # save full matrix
            self._ssim_mat_fp = ssim_mats_array
        return float(self._ssim_value_fp_fast_exp)

    @property
    def ssim_value_fp_orig(self):
        """the ssim on the fp data with
        original constants and no scaling (so-called "straightforward" approach.
        This will return Nan on POP data or CAM data with NaNs because scikit
        SSIM fuction does not handle NaNs.
        """

        import numpy as np
        from skimage.metrics import structural_similarity as ssim

        if not self._is_memoized('_ssim_value_fp_orig'):

            # if this is a 3D variable, we will do each level seperately
            if self._calcs1._vert_dim_name is not None:
                vname = self._calcs1._vert_dim_name
                if vname not in self._calcs1.get_calc('ds').sizes:
                    nlevels = 1
                else:
                    nlevels = self._calcs1.get_calc('ds').sizes[vname]
            else:
                nlevels = 1

            ssim_levs = np.zeros(nlevels)

            for this_lev in range(nlevels):
                if nlevels == 1:
                    a1 = self._calcs1.get_calc('ds').data
                    a2 = self._calcs2.get_calc('ds').data
                else:
                    a1 = self._calcs1.get_calc('ds').isel({vname: this_lev}).data
                    a2 = self._calcs2.get_calc('ds').isel({vname: this_lev}).data

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
                ssim_levs[this_lev] = mean_ssim

            # end of levels calculation
            return_ssim = ssim_levs.min()

            self._ssim_value_fp_orig = return_ssim

        return self._ssim_value_fp_orig

    def get_diff_calc(self, name: str, color: Optional[str] = 'coolwarm', xsize=11, ysize=11):
        """
        Gets a calc on the dataset that requires more than one input dataset

        Parameters
        ==========
        name : str
            The name of the calc (must be identical to a property name)

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
            if name == 'spre_tol':
                return self.spre_tol
            if name == 'ssim':
                self.color = color
                return self.ssim_value
            if name == 'ssim_fp':
                self.xsize = xsize
                self.ysize = ysize
                return self.ssim_value_fp_fast
            if name == 'ssim_fp_exp':
                return self.ssim_value_fp_fast_exp
            if name == 'ssim_fp_orig':  # this is using standard SSIM with floats
                # ("straightforward approach")
                # not recommended
                return self.ssim_value_fp_orig
            if name == 'ssim_fp_slow':
                # the non-matrix DSSIM implementation - good for experimenting
                # not recommended in practice
                return self.ssim_value_fp_slow
            raise ValueError(f'there is no calc with the name: {name}.')
        else:
            raise TypeError('name must be a string.')
