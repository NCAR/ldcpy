import calendar
import datetime
import math
import re

import cmocean
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
        set1,
        metric,
        set2=None,
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
        self._set1_name = set1
        self._set2_name = set2
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
        da_data = da
        da_data.attrs = da.attrs
        if self._metric_type == 'diff' and self._standardized_err is True:
            if da.std(dim='time').all() == 0:
                da_attrs = da.attrs
                da_data = (da - da.mean(dim='time')) / da.std(dim='time')
                da_data.attrs = da_attrs
            else:
                raise ValueError(
                    'Standard deviation of error data is 0. Cannot standardize errors.'
                )

        if self._plot_type in ['spatial', 'spatial_comparison']:
            metrics_da = lm.DatasetMetrics(da_data, ['time'])
        elif self._plot_type in ['time_series', 'periodogram', 'histogram']:
            metrics_da = lm.DatasetMetrics(da_data, ['lat', 'lon'])
        else:
            raise ValueError(f'plot type {self._plot_type} not supported')

        raw_data = metrics_da.get_metric(self._metric)
        return raw_data

    def get_plot_data(self, raw_data_1, raw_data_2=None):
        if self._metric_type == 'diff':
            plot_data = raw_data_1 - raw_data_2
            plot_data.attrs = raw_data_1.attrs
        elif self._metric_type == 'ratio':
            plot_data = raw_data_2 / raw_data_1
            plot_data.attrs = raw_data_1.attrs
            if hasattr(self._ds, 'units'):
                self._odds_positive.attrs['units'] = ''

        elif self._metric_type == 'raw' or self._metric_type == 'metric_of_diff':
            plot_data = raw_data_1
        else:
            raise ValueError(f'metric_type {self._metric_type} not supported')

        if self._group_by is not None:
            plot_attrs = plot_data.attrs
            plot_data = plot_data.groupby(self._group_by).mean(dim='time')
            plot_data.attrs = plot_attrs

        if self._transform == 'none':
            pass
        elif self._transform == 'log':
            plot_attrs = plot_data.attrs
            plot_data = np.log10(plot_data)
            plot_data.attrs = plot_attrs
        else:
            raise ValueError(f'metric transformation {self._transform} not supported')

        return plot_data

    def get_title(self, metric_name, c_name=None):

        if self._set2_name is not None and self._plot_type != 'spatial_comparison':
            das = f'{self._set1_name}, {self._set2_name}'
        elif c_name is not None:
            das = f'{c_name}'
        else:
            das = f'{self._set1_name}'

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

    def _label_offset(
        self, ax,
    ):
        fmt = ax.yaxis.get_major_formatter()
        ax.yaxis.offsetText.set_visible(False)
        set_label = ax.set_ylabel
        label = ax.get_ylabel()

        def update_label(event_axes):
            offset = fmt.get_offset()
            if offset == '':
                set_label('{}'.format(label))
            else:
                set_label('{} ({})'.format(label, offset))
            return

        ax.callbacks.connect('ylim_changed', update_label)
        ax.figure.canvas.draw()
        update_label(None)
        return

    def spatial_comparison_plot(self, da_set1, title_set1, da_set2, title_set2):
        lat_set1 = da_set1['lat']
        lat_set2 = da_set2['lat']
        cy_data_set1, lon_set1 = add_cyclic_point(da_set1, coord=da_set1['lon'])
        cy_data_set2, lon_set2 = add_cyclic_point(da_set2, coord=da_set2['lon'])

        fig = plt.figure(dpi=300, figsize=(9, 2.5))

        mymap = plt.get_cmap(f'{self._color}')
        mymap.set_under(color='black')
        mymap.set_over(color='white')
        mymap.set_bad(alpha=0)

        ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=0.0))

        ax1.set_facecolor('#39ff14')
        ax1.set_title(title_set1)

        no_inf_data_set1 = np.nan_to_num(cy_data_set1, nan=np.nan)
        color_min = min(
            np.min(da_set1.where(da_set1 != -inf)).values.min(),
            np.min(da_set2.where(da_set2 != -inf)).values.min(),
        )
        color_max = max(
            np.max(da_set1.where(da_set1 != inf)).values.max(),
            np.max(da_set2.where(da_set2 != inf)).values.max(),
        )
        pset2 = ax1.pcolormesh(
            lon_set1,
            lat_set1,
            no_inf_data_set1,
            transform=ccrs.PlateCarree(),
            cmap=mymap,
            vmin=color_min,
            vmax=color_max,
        )
        ax1.set_global()
        ax1.coastlines()

        ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=0.0))

        ax2.set_facecolor('#39ff14')
        ax2.set_title(title_set2)

        no_inf_data_set2 = np.nan_to_num(cy_data_set2, nan=np.nan)
        pc2 = ax2.pcolormesh(
            lon_set2,
            lat_set2,
            no_inf_data_set2,
            transform=ccrs.PlateCarree(),
            cmap=mymap,
            vmin=color_min,
            vmax=color_max,
        )

        ax2.set_global()
        ax2.coastlines()

        # add colorbar
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
        cax = fig.add_axes([0.1, 0, 0.8, 0.05])

        if not (np.isnan(cy_data_set1).all() and np.isnan(cy_data_set2).all()):
            if np.isinf(cy_data_set1).any() or np.isinf(cy_data_set2).any():
                fig.colorbar(pset2, cax=cax, orientation='horizontal', shrink=0.95, extend='both')
                cb = fig.colorbar(
                    pc2, cax=cax, orientation='horizontal', shrink=0.95, extend='both'
                )
                cb.ax.set_title(f'{da_set1.units}')
            else:
                fig.colorbar(pset2, cax=cax, orientation='horizontal', shrink=0.95)
                cb = fig.colorbar(pc2, cax=cax, orientation='horizontal', shrink=0.95)
                cb.ax.set_title(f'{da_set1.units}')
            cb.ax.tick_params(labelsize=8, rotation=30)
        else:
            proxy = [plt.Rectangle((0, 0), 1, 1, fc='#39ff14')]
            plt.legend(proxy, ['NaN'])

    def spatial_plot(self, da, title):

        lat = da['lat']
        cy_data, lon = add_cyclic_point(da, coord=da['lon'],)

        mymap = plt.get_cmap(self._color)
        mymap.set_under(color='black')
        mymap.set_over(color='white')
        mymap.set_bad(alpha=0.0)
        ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0.0))

        ax.set_facecolor('#39ff14')

        masked_data = np.nan_to_num(cy_data, nan=np.nan)
        color_min = np.min(da.where(da != -inf))
        color_max = np.max(da.where(da != inf))
        pc = ax.pcolormesh(
            lon,
            lat,
            masked_data,
            transform=ccrs.PlateCarree(),
            cmap=mymap,
            vmin=color_min.values.min(),
            vmax=color_max.values.max(),
        )
        if not np.isnan(cy_data).all():
            if np.isinf(cy_data).any():
                cb = plt.colorbar(pc, orientation='horizontal', shrink=0.95, extend='both')
            else:
                cb = plt.colorbar(pc, orientation='horizontal', shrink=0.95)
            cb.ax.tick_params(labelsize=8, rotation=30)
            cb.ax.set_title(f'{da.units}')
        else:
            proxy = [plt.Rectangle((0, 0), 1, 1, fc='#39ff14')]
            plt.legend(proxy, ['NaN'])

        ax.set_global()
        ax.coastlines()
        ax.set_title(title)

    def hist_plot(self, plot_data, title):
        fig, axs = mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(plot_data)
        if plot_data.units != '':
            mpl.pyplot.xlabel(f'{self._metric} ({plot_data.units})')
        else:
            mpl.pyplot.xlabel(f'{self._metric}')
        mpl.pyplot.title(f'time-series histogram: {title}')

    def periodogram_plot(self, plot_data, title):
        dat = xrft.dft((plot_data - plot_data.mean()).chunk((plot_data - plot_data.mean()).size))
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
        tick_interval = int(da.size / 5) + 1
        if da.size == 1:
            tick_interval = 1
        if self._group_by == 'time.dayofyear':
            group_string = 'dayofyear'
            xlabel = 'Day of Year'
        elif self._group_by == 'time.month':
            group_string = 'month'
            xlabel = 'Month'
        elif self._group_by == 'time.year':
            group_string = 'year'
            xlabel = 'Year'
        elif self._group_by == 'time.day':
            group_string = 'day'
            xlabel = 'Day'

        if self._metric_type == 'diff':
            if da.units != '':
                ylabel = f'{self._metric} ({da.units}) diff'
            else:
                ylabel = f'{self._metric} diff'
        elif self._metric_type == 'ratio':
            ylabel = f'ratio {self._metric}'
        else:
            if da.units != '':
                ylabel = f'{self._metric} ({da.units})'
            else:
                ylabel = f'{self._metric} diff'

        if self._transform == 'log':
            plot_ylabel = f'log10({ylabel})'
        else:
            plot_ylabel = ylabel

        if self._group_by is not None:
            mpl.pyplot.plot(da[group_string].data, da, 'bo')
            ax = mpl.pyplot.gca()
        else:
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
            plt.gca().xaxis.set_major_locator(mdates.DayLocator())
            dtindex = da.indexes['time'].to_datetimeindex()
            da['time'] = dtindex

            mpl.pyplot.plot_date(da.time.data, da, 'bo')
            ax = mpl.pyplot.gca()

        mpl.pyplot.ylabel(plot_ylabel)
        mpl.pyplot.yscale(self._scale)
        self._label_offset(ax)
        mpl.pyplot.xlabel(xlabel)

        if self._group_by == 'time.month':
            int_labels = [item.get_text() for item in ax.get_xticklabels()]
            no_hyphen_int_labels = [(int(float(re.sub('−', '-', label)))) for label in int_labels]
            month_labels = [
                calendar.month_name[no_hyphen_int_label]
                if calendar.month_name[no_hyphen_int_label] != ''
                else ''
                for no_hyphen_int_label in no_hyphen_int_labels
            ]
            unique_month_labels = list(dict.fromkeys(month_labels))
            plt.gca().set_xticklabels(unique_month_labels)

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
    metric,
    set1,
    set2=None,
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
    metric -- string
        the name of the metric to be plotted (must match a property name in the DatasetMetrics class in ldcpy.plot, for more information about the available metrics see ldcpy.DatasetMetrics)

            'ns_con_var'

            'ew_con_var'

            'mean'

            'std'

            'variance'

            'prob_positive'

            'prob_negative'

            'odds_positive'

            'zscore'

            'mean_abs'

            'mean_squared'

            'rms'

            'sum'

            'sum_squared'

            'corr_lag1'

            'quantile'

            'lag1'


    set1 -- string
        the label of the dataset to gather metrics from

    Keyword Arguments:
    ==================
    set2 -- string
        the label of the second dataset to gather metrics from (needed if metric_type is diff, ratio, or metric_of_diff, or if plot_type is spatial_comparison)

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
        The type of operation to be performed on the metrics in the two collections. Valid options:

            'raw': the unaltered metric values

            'diff': the difference between the metric values in collections set1 and set2

            'ratio': the ratio of the metric values in (set2/set1)

            'metric_of_diff': the metric value computed on the difference between set1 and set2

    plot_type -- string (default 'spatial')
        The type of plot to be created. Valid options:

            'spatial': a plot of the world with values at each lat and lon point (takes the mean across the time dimension)

            'spatial_comparison': two side-by-side spatial plots, one of the raw metric from set1 and the other of the raw metric from set2

            'time-series': A time-series plot of the data (computed by taking the mean across the lat and lon dimensions)

            'histogram': A histogram of the time-series data

    transform -- string (default 'none')
        data transformation. Valid options:

            'none'

            'log'

    subset -- string (default None)
        subset of the data to gather metrics on. Valid options:

            'first5': the first 5 days of data

            'winter': data from the months December, January, February

            'spring': data from the months March, April, May

            'summer': data from the months June, July, August

            'autumn': data from the months September, October, November

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
        set1,
        metric,
        set2,
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
    subset_set1 = lu.subset_data(
        ds[varname].sel(collection=set1), subset, lat, lon, lev, start, end
    )
    if set2 is not None:
        subset_set2 = lu.subset_data(
            ds[varname].sel(collection=set2), subset, lat, lon, lev, start, end
        )

    # Acquire raw metric values
    if metric_type in ['metric_of_diff']:
        data = subset_set1 - subset_set2
        data.attrs = subset_set1.attrs
    else:
        data = subset_set1

    raw_metric_set1 = mp.get_metrics(data)
    # TODO: This will plot a second plot even if metric_type is metric_of diff in spatial comparison case
    if plot_type in ['spatial_comparison'] or metric_type in ['diff', 'ratio']:
        raw_metric_set2 = mp.get_metrics(subset_set2)

    # Get metric names/values for plot title
    # if metric == 'zscore':
    metric_name_set1 = mp.get_metric_label(metric, data, ds['gw'].values)
    if metric == 'mean' and plot_type == 'spatial_comparison':
        metric_name_set2 = mp.get_metric_label(metric, subset_set2, ds['gw'].values)

    # Get plot data and title
    if lat is not None and lon is not None:
        mp.title_lat = subset_set1['lat'].data[0]
        mp.title_lon = subset_set1['lon'].data[0] - 180
    else:
        mp.title_lat = lat
        mp.title_lon = lon

    if metric_type in ['diff', 'ratio']:
        plot_data_set1 = mp.get_plot_data(raw_metric_set1, raw_metric_set2)
    else:
        plot_data_set1 = mp.get_plot_data(raw_metric_set1)
    title_set1 = mp.get_title(metric_name_set1, set1)
    if plot_type == 'spatial_comparison':
        plot_data_set2 = mp.get_plot_data(raw_metric_set2)
        title_set2 = mp.get_title(metric_name_set1, set2)
        if metric == 'mean':
            title_set2 = mp.get_title(metric_name_set2, set2)

    # Call plot functions
    if plot_type == 'spatial_comparison':
        mp.spatial_comparison_plot(plot_data_set1, title_set1, plot_data_set2, title_set2)
    elif plot_type == 'spatial':
        mp.spatial_plot(plot_data_set1, title_set1)
    elif plot_type == 'time_series':
        mp.time_series_plot(plot_data_set1, title_set1)
    elif plot_type == 'histogram':
        mp.hist_plot(plot_data_set1, title_set1)
    elif plot_type == 'periodogram':
        mp.periodogram_plot(plot_data_set1, title_set1)
