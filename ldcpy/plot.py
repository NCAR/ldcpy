import math

import matplotlib as mpl
import numpy as np
import pandas as pd
import xrft
from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib import dates as mdates
from matplotlib import pyplot as plt
from numpy import inf

from ldcpy import metrics as lm
from ldcpy import util as lu


class MetricsPlot(object):
    """
    This class contains code to plot metrics in an xarray Dataset that has either 'lat' and 'lon' dimensions, or a
    'time' dimension.
    """

    def __init__(
        self,
        ds,
        varname,
        ens_o,
        metric,
        ens_r=None,
        group_by=None,
        scale='linear',
        metric_type='raw',
        plot_type='spatial',
        transform='none',
        subset=None,
        approx_lat=None,
        approx_lon=None,
        lev=0,
        color='coolwarm',
        standardized_err=False,
        quantile=0.5,
        contour_levs=24,
    ):

        self._ds = ds

        # Metric settings used in plot titles
        self._varname = varname
        self._ens_o_name = ens_o
        self._ens_r_name = ens_r
        self._title_lat = None
        self._title_lon = None

        # Plot settings
        self._metric = metric
        self._group_by = group_by
        self._scale = scale
        self._metric_type = metric_type
        self._plot_type = plot_type
        self._subset = subset
        self._true_lat = approx_lat
        self._true_lon = approx_lon
        self._transform = transform
        self._lev = lev
        self._color = color
        self._standardized_err = standardized_err
        self._quantile = quantile
        self._contour_levs = contour_levs

    def get_metrics(self, da):
        if self._plot_type in ['spatial', 'spatial_comparison']:
            metrics_da = lm.DatasetMetrics(da, ['time'])
        elif self._plot_type in ['time_series', 'periodogram', 'histogram']:
            metrics_da = lm.DatasetMetrics(da, ['lat', 'lon'])
        else:
            raise ValueError(f'plot type {self._plot_type} not supported')

        raw_data = metrics_da.get_metric(self._metric)

        return raw_data

    def get_plot_data(self, raw_data_1, raw_data_2=None):
        if self._metric_type == 'diff':
            plot_data = raw_data_1 - raw_data_2
        elif self._metric_type == 'ratio':
            plot_data = raw_data_2 / raw_data_1
        elif self._metric_type == 'raw' or self._metric_type == 'metric_of_diff':
            plot_data = raw_data_1
        else:
            raise ValueError(f'metric_type {self._metric_type} not supported')

        if self._metric_type == 'diff' and self._standardized_err is True:
            if plot_data.std(dim='time') != 0:
                plot_data = (plot_data - plot_data.mean(dim='time')) / plot_data.std(dim='time')
            else:
                raise ValueError(
                    'Standard deviation of error data is 0. Cannot standardize errors.'
                )

        if self._group_by is not None:
            plot_data = plot_data.groupby(self._group_by).mean(dim='time')

        if self._transform == 'none':
            pass
        elif self._transform == 'log':
            plot_data = np.log10(plot_data)
        else:
            raise ValueError(f'metric transformation {self._transform} not supported')

        return plot_data

    def get_title(self, metric_name):

        if self._ens_r_name is not None:
            das = f'{self._ens_o_name}, {self._ens_r_name}'
        else:
            das = f'{self._ens_o_name}'

        if self._quantile is not None and metric_name == 'quantile':
            metric_full_name = f'{metric_name} {self._quantile}'
        else:
            metric_full_name = metric_name

        if self._transform == 'log':
            title = f'{das}: {self._varname}: log10({metric_full_name})'
        else:
            title = f'{das}: {self._varname}: {metric_full_name}'

        if self._metric_type != 'raw':
            title = f'{title} {self._metric_type}'

        if self._group_by is not None:
            title = f'{title} by {self._group_by}'

        if self.title_lat is not None:
            if self.title_lon is not None:
                title = f'{title} at lat={self.title_lat:.2f}, lon={self.title_lon:.2f}'
            else:
                title = f'{title} at lat={self.title_lat:.2f}'
        elif self.title_lon is not None:
            title = f'{title} at lat={self.title_lon:.2f}'

        if self._subset is not None:
            title = f'{title} subset:{self._subset}'

        return title

    def _calc_contour_levels(self, dat_1, dat_2=None):
        """
        TODO: minval returns the smallest value not equal to -inf, is there a more elegant solution to plotting -inf values
        (for EW contrast variance in particular)?
        """
        # both plots use same contour levels
        if dat_2 is not None:
            minval = np.nanmin(np.minimum(dat_1, dat_2))
            maxval = np.nanmax(np.maximum(dat_1, dat_2))
            if minval == -math.inf:
                if np.isfinite(dat_1).any() or np.isfinite(dat_2).any():
                    minval = np.minimum(
                        dat_1.where(dat_1 != -inf).min(), dat_2.where(dat_2 != -inf).min()
                    ).data
                else:
                    return np.array([0, 0.00000001])
            if maxval == math.inf:
                if np.isfinite(dat_1).any() or np.isfinite(dat_2).any():
                    maxval = np.maximum(
                        dat_1.where(dat_1 != -inf).max(), dat_2.where(dat_2 != -inf).max()
                    ).data
                else:
                    return np.array([0, 0.00000001])
        else:
            minval = np.nanmin(dat_1)
            maxval = np.nanmax(dat_1)
            if minval == -math.inf:
                if np.isfinite(dat_1).any():
                    minval = dat_1.where(dat_1 != -inf).min().data
                else:
                    return np.array([0, 0.00000001])
            if maxval == math.inf:
                if np.isfinite(dat_1).any():
                    maxval = dat_1.where(dat_1 != inf).max().data
                else:
                    return np.array([0, 0.00000001])
        levels = minval + np.arange(self._contour_levs + 1) * (maxval - minval) / self._contour_levs
        # print('Min value: {}\nMax value: {}'.format(minval, maxval))
        return levels

    def spatial_comparison_plot(self, da_o, title_o, da_r, title_r):
        lat_o = da_o['lat']
        lat_r = da_r['lat']
        if (da_o == inf).all():
            cy_data_o, lon_o = add_cyclic_point(da_o.where(da_o != inf, 1), coord=da_o['lon'])
        elif (da_o == -inf).all():
            cy_data_o, lon_o = add_cyclic_point(da_o.where(da_o != -inf, -1), coord=da_o['lon'])
        else:
            cy_data_o, lon_o = add_cyclic_point(
                da_o.where(da_o != inf, da_o.where(da_o != inf).max() + 1).where(
                    da_o != -inf, da_o.where(da_o != -inf).min() - 1
                ),
                coord=da_o['lon'],
            )

        if (da_r == inf).all():
            cy_data_r, lon_r = add_cyclic_point(da_r.where(da_r != inf, 1), coord=da_r['lon'])
        elif (da_r == -inf).all():
            cy_data_r, lon_r = add_cyclic_point(da_r.where(da_r != -inf, -1), coord=da_r['lon'])
        else:
            cy_data_r, lon_r = add_cyclic_point(
                da_r.where(da_r != inf, da_r.where(da_r != inf).max() + 1).where(
                    da_r != -inf, da_r.where(da_r != -inf).min() - 1
                ),
                coord=da_r['lon'],
            )

        # cy_data_r, lon_r = add_cyclic_point(da_r.where(da_r!=-inf, -sys.float_info.max).where(da_r!=inf, sys.float_info.max), coord=da_r['lon'])
        fig = plt.figure(dpi=300, figsize=(9, 2.5))

        mymap = plt.get_cmap(f'{self._color}')
        mymap.set_under(color='black')
        mymap.set_over(color='white')
        mymap.set_bad(alpha=0)

        # both plots use same contour levels
        levels = self._calc_contour_levels(da_o, da_r)

        ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=0.0))
        ax1.set_title(title_o)

        if (levels == levels[0]).all():
            pc1 = ax1.pcolormesh(lon_o, lat_o, cy_data_o, transform=ccrs.PlateCarree(), cmap=mymap)
        else:
            if np.isfinite(da_o).all() and np.isfinite(da_r).all():
                pc1 = ax1.contourf(
                    lon_o,
                    lat_o,
                    cy_data_o,
                    transform=ccrs.PlateCarree(),
                    cmap=mymap,
                    levels=levels,
                    extend='neither',
                )
            else:
                pc1 = ax1.contourf(
                    lon_o,
                    lat_o,
                    cy_data_o,
                    transform=ccrs.PlateCarree(),
                    cmap=mymap,
                    levels=levels,
                    extend='both',
                )
        ax1.set_global()
        ax1.coastlines()

        ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=0.0))
        ax2.set_title(title_r)
        if (levels == levels[0]).all():
            pc2 = ax2.pcolormesh(lon_r, lat_r, cy_data_r, transform=ccrs.PlateCarree(), cmap=mymap)
        else:
            if np.isfinite(da_o).all() and np.isfinite(da_r).all():
                pc2 = ax2.contourf(
                    lon_r,
                    lat_r,
                    cy_data_r,
                    transform=ccrs.PlateCarree(),
                    cmap=mymap,
                    levels=levels,
                    extend='neither',
                )
            else:
                pc2 = ax1.contourf(
                    lon_o,
                    lat_o,
                    cy_data_o,
                    transform=ccrs.PlateCarree(),
                    cmap=mymap,
                    levels=levels,
                    extend='both',
                )
        ax2.set_global()
        ax2.coastlines()

        # add colorbar
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
        cax = fig.add_axes([0.1, 0, 0.8, 0.05])
        if not (np.isnan(pc1.levels).all() and np.isnan(pc2.levels).all()):
            cbar = fig.colorbar(pc1, cax=cax, orientation='horizontal')
            cbar = fig.colorbar(pc2, cax=cax, orientation='horizontal')
            cbar.ax.tick_params(labelsize=8, rotation=30)
        # cbar.ax.set_xticklabels(['{:.2f}'.format(i) for i in cbar.get_ticks()])

    def spatial_plot(self, da, title):
        """
        visualize the mean error
        want to be able to input multiple?
        """

        levels = self._calc_contour_levels(da)

        lat = da['lat']
        if (da == inf).all():
            cy_data, lon = add_cyclic_point(da.where(da != inf, 1), coord=da['lon'])
        elif (da == -inf).all():
            cy_data, lon = add_cyclic_point(da.where(da != inf, -1), coord=da['lon'])
        else:
            cy_data, lon = add_cyclic_point(
                da.where(da != inf, da.where(da != inf).max() + 1).where(
                    da != -inf, da.where(da != -inf).min() - 1
                ),
                coord=da['lon'],
            )

        mymap = plt.get_cmap(self._color)
        mymap.set_under(color='black')
        mymap.set_over(color='white')
        mymap.set_bad(alpha=0)
        # cy_data = np.ma.masked_invalid(cy_data)
        ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0.0))
        # pc = ax.pcolormesh(lon, lat, cy_data, transform=ccrs.PlateCarree(), cmap=mymap)

        # if xr.ufuncs.isnan(cy_data).any():
        ax.set_facecolor('gray')

        if (levels == levels[0]).all():
            pc = ax.pcolormesh(lon, lat, cy_data, transform=ccrs.PlateCarree(), cmap=mymap)
        else:
            if np.isfinite(da).all():
                pc = ax.contourf(
                    lon,
                    lat,
                    cy_data,
                    transform=ccrs.PlateCarree(),
                    cmap=mymap,
                    levels=levels,
                    extend='neither',
                )
            else:
                pc = ax.contourf(
                    lon,
                    lat,
                    cy_data,
                    transform=ccrs.PlateCarree(),
                    cmap=mymap,
                    levels=levels,
                    extend='both',
                )
        if not np.isnan(pc.levels).all():
            cb = plt.colorbar(pc, orientation='horizontal', shrink=0.95)
            cb.ax.tick_params(labelsize=8, rotation=30)
        else:
            proxy = [plt.Rectangle((0, 0), 1, 1, fc='gray')]
            plt.legend(proxy, ['NaN'])

        ax.set_global()
        ax.coastlines()
        ax.set_title(title)

    def hist_plot(self, plot_data, title):
        fig, axs = mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(plot_data)
        mpl.pyplot.xlabel(self._metric)
        mpl.pyplot.title(f'time-series histogram: {title}')

    def periodogram_plot(self, plot_data, title):
        dat = xrft.dft(plot_data - plot_data.mean())
        i = (np.multiply(dat, np.conj(dat)) / dat.size).real
        i = np.log10(i[2 : int(dat.size / 2) + 1])
        freqs = np.array(range(1, int(dat.size / 2))) / dat.size

        mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
        mpl.pyplot.plot(freqs, i)
        mpl.pyplot.title(f'periodogram: {title}')

    def time_series_plot(
        self, da, title,
    ):
        """
        time series plot
        """
        group_string = 'time.year'
        xlabel = 'date'
        tick_interval = 28
        if self._group_by == 'time.dayofyear':
            group_string = 'dayofyear'
            xlabel = 'Day of Year'
            tick_interval = 20
        elif self._group_by == 'time.month':
            group_string = 'month'
            xlabel = 'Month'
            tick_interval = 1
        elif self._group_by == 'time.year':
            group_string = 'year'
            xlabel = 'Year'
            tick_interval = 1
        elif self._group_by == 'time.day':
            group_string = 'day'
            xlabel = 'Day'
            tick_interval = 20

        if self._metric_type == 'diff':
            ylabel = f'{self._metric} error'
        elif self._metric_type == 'ratio':
            ylabel = f'ratio {self._metric}'
        else:
            ylabel = f'{self._metric}'

        if self._transform == 'log':
            plot_ylabel = f'log10({ylabel})'
        else:
            plot_ylabel = ylabel

        if self._group_by is not None:
            mpl.pyplot.plot(da[group_string].data, da, 'bo')
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            dtindex = da.indexes['time'].to_datetimeindex()
            da['time'] = dtindex
            mpl.pyplot.plot_date(da.time.data, da, 'bo')

        mpl.pyplot.ylabel(plot_ylabel)
        mpl.pyplot.yscale(self._scale)
        mpl.pyplot.xlabel(xlabel)

        if self._group_by is not None:
            mpl.pyplot.xticks(
                np.arange(min(da[group_string]), max(da[group_string]) + 1, tick_interval)
            )
        else:
            mpl.pyplot.xticks(
                pd.date_range(
                    np.datetime64(da['time'][0].data),
                    np.datetime64(da['time'][-1].data),
                    periods=int(da['time'].size / tick_interval) + 1,
                )
            )

        mpl.pyplot.title(title)

    def get_metric_label(self, metric, data, weights=None):
        # Get special metric names
        if metric == 'zscore':
            zscore_cutoff = lm.DatasetMetrics((data), ['time']).get_single_metric('zscore_cutoff')
            percent_sig = lm.DatasetMetrics((data), ['time']).get_single_metric(
                'zscore_percent_significant'
            )
            metric_name = f'{metric}: cutoff {zscore_cutoff[0]:.2f}, % sig: {percent_sig:.2f}'
        elif metric == 'mean' and self._plot_type == 'spatial_comparison':
            o_wt_mean = np.average(
                np.average(
                    lm.DatasetMetrics(data, ['time']).get_metric(metric), axis=0, weights=weights,
                )
            )
            metric_name = f'{metric} = {o_wt_mean:.2f}'
        else:
            metric_name = metric

        return metric_name


def plot(
    ds,
    varname,
    ens_o,
    metric,
    ens_r=None,
    group_by=None,
    scale='linear',
    metric_type='raw',
    plot_type='spatial',
    transform='none',
    subset=None,
    lat=None,
    lon=None,
    lev=0,
    color='coolwarm',
    standardized_err=False,
    quantile=0.5,
    start=None,
    end=None,
):
    """
    Plots the data given an xarray dataset


    Parameters:
    ===========
    ds -- xarray.Dataset
        the dataset
    varname -- string
        the name of the variable to be plotted
    ens_o -- string
        the ensemble name of the dataset to gather metrics from
    metric -- string
        the name of the metric to be plotted (must match a property name in the DatasetMetrics class in ldcpy.plot)

    Keyword Arguments:
    ==================
    ens_r -- string
        the name of the second dataset to gather metrics from (needed if metric_type is diff, ratio, or metric_of_diff, or if plot_type is spatial_comparison)

    group_by -- string
        how to group the data in time series plots. Valid groupings:
            "time.day"

            "time.dayofyear"

            "time.month"

            "time.year"

    scale -- string (default "linear")
        time-series y-axis plot transformation. Valid options:
            'linear'

            'log'

    metric_type -- string (default 'raw')
        The type of operation to be performed on the metrics in the two ensembles. Valid options:

            'raw': the unaltered metric values

            'diff': the difference between the metric values in ens_o and ens_r

            'ratio': the ratio of the metric values in (ens_r/ens_o)

            'metric_of_diff': the metric value computed on the difference between ens_o and ens_r

    plot_type -- string (default 'spatial')
        The type of plot to be created. Valid options:

            'spatial': a plot of the world with values at each lat and lon point (takes the mean across the time dimension)

            'spatial_comparison': two side-by-side spatial plots, one of the raw metric from ens_o and the other of the raw metric from ens_r

            'time-series': A time-series plot of the data (computed by taking the mean across the lat and lon dimensions)

            'histogram': A histogram of the time-series data

    transform -- string (default 'none')
        data transformation. Valid options:

            'none'

            'log'

    subset -- string (default None)
        subset of the data to gather metrics on. Valid options:

            'first50': the first 50 days of data

            'winter': data from the months December, January, February

    lat -- float (default None)
        the latitude of the data to gather metrics on.

    lon -- float (default None)
        the longitude of the data to gather metrics on.

    lev -- float (default 0)
        the level of the data to gather metrics on (used if plotting from a 3d data set).

    color -- string (default 'coolwarm')
        the color scheme for spatial plots (see https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html)

    standardized_err -- bool (default False)
        whether or not to standardize the error in a plot of metric_type="diff"

    quantile -- float (default 0.5)
        a value between 0 and 1 required if metric="quantile", corresponding to the desired quantile to gather


    start -- int (default None)
        a value between 0 and the number of time slices indicating the start time of a subset


    end -- int (default None)
        a value between 0 and the number of time slices indicating the end time of a subset

    Returns
    =======
    out -- None
    """

    mp = MetricsPlot(
        ds,
        varname,
        ens_o,
        metric,
        ens_r,
        group_by,
        scale,
        metric_type,
        plot_type,
        transform,
        subset,
        lat,
        lon,
        lev,
        color,
        standardized_err,
        quantile,
    )

    # Subset data
    subset_o = lu.subset_data(ds[varname].sel(ensemble=ens_o), subset, lat, lon, lev, start, end)
    if ens_r is not None:
        subset_r = lu.subset_data(
            ds[varname].sel(ensemble=ens_r), subset, lat, lon, lev, start, end
        )

    # Acquire raw metric values
    if metric_type in ['metric_of_diff']:
        data = subset_o - subset_r
    else:
        data = subset_o

    raw_metric_o = mp.get_metrics(data)
    # TODO: This will plot a second plot even if metric_type is metric_of diff in spatial comparison case
    if plot_type in ['spatial_comparison'] or metric_type in ['diff', 'ratio']:
        raw_metric_r = mp.get_metrics(subset_r)

    # Get metric names/values for plot title
    # if metric == 'zscore':
    metric_name_o = mp.get_metric_label(metric, data, ds['gw'].values)
    if metric == 'mean' and plot_type == 'spatial_comparison':
        metric_name_r = mp.get_metric_label(metric, subset_r, ds['gw'].values)

    # Get plot data and title
    if lat is not None and lon is not None:
        mp.title_lat = subset_o['lat'].data[0]
        mp.title_lon = subset_o['lon'].data[0] - 180
    else:
        mp.title_lat = lat
        mp.title_lon = lon

    if metric_type in ['diff', 'ratio']:
        plot_data_o = mp.get_plot_data(raw_metric_o, raw_metric_r)
    else:
        plot_data_o = mp.get_plot_data(raw_metric_o)
    title_o = mp.get_title(metric_name_o)
    if plot_type == 'spatial_comparison':
        plot_data_r = mp.get_plot_data(raw_metric_r)
        title_r = mp.get_title(metric_name_o)
        if metric == 'mean':
            title_r = mp.get_title(metric_name_r)

    # Call plot functions
    if plot_type == 'spatial_comparison':
        mp.spatial_comparison_plot(plot_data_o, title_o, plot_data_r, title_r)
    elif plot_type == 'spatial':
        mp.spatial_plot(plot_data_o, title_o)
    elif plot_type == 'time_series':
        mp.time_series_plot(plot_data_o, title_o)
    elif plot_type == 'histogram':
        mp.hist_plot(plot_data_o, title_o)
    elif plot_type == 'periodogram':
        mp.periodogram_plot(plot_data_o, title_o)
