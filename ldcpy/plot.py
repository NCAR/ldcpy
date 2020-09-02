import calendar
import copy
import warnings

import cmocean
import matplotlib as mpl
import nc_time_axis
import numpy as np
import pandas as pd
import xarray as xr
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
        metric,
        sets,
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
        quantile=None,
        calc_ssim=False,
        contour_levs=24,
    ):

        self._ds = ds

        # Metric settings used in plot titles
        self._varname = varname
        self._sets = sets
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
        self._quantile = None
        self._calc_ssim = calc_ssim
        self._contour_levs = contour_levs

    def verify_plot_parameters(self):
        if len(self._sets) < 2 and self._metric_type in [
            'diff',
            'ratio',
            'metric_of_diff',
        ]:
            raise ValueError(f'Must specify set2 for {self._metric_type} metric type')
        if self._plot_type in ['spatial'] and self._group_by is not None:
            raise ValueError(f'Cannot group by {self._group_by} in a non-time-series plot')
        if self._plot_type not in ['spatial'] and self._color != 'coolwarm':
            raise ValueError('Cannot change color scheme in a non-spatial plot')
        if self._plot_type in ['spatial'] and (
            self._true_lat is not None or self._true_lon is not None
        ):
            raise ValueError('Cannot currently subset by latitude or longitude in a spatial plot')
        if self._standardized_err is not False and self._metric_type != 'diff':
            raise ValueError("Cannot standardize errors if metric_type != 'diff'")
        if self._lev != 0 and 'lev' not in self._ds.dims:
            raise ValueError('Cannot subset by lev in this dataset')
        if self._quantile is not None and self._metric != 'quantile':
            raise ValueError('Cannot change quantile value if metric is not quantile')
        if self._quantile is None and self._metric == 'quantile':
            raise ValueError('Must specify quantile value as argument')

        if self._metric in ['lag1', 'corr_lag1', 'mae_day_max'] and self._plot_type not in [
            'spatial',
        ]:
            raise ValueError(f'Cannot plot {self._metric} in a non-spatial plot')

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

        if self._plot_type in ['spatial']:
            metrics_da = lm.DatasetMetrics(da_data, ['time'])
        elif self._plot_type in ['time_series', 'periodogram', 'histogram']:
            metrics_da = lm.DatasetMetrics(da_data, ['lat', 'lon'])
        else:
            raise ValueError(f'plot type {self._plot_type} not supported')

        if self._calc_ssim and self._plot_type != 'spatial':
            warnings.warn(
                'SSIM is only calculated for spatial plots, ignoring calc_ssim option', UserWarning
            )

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

        if c_name is not None:
            das = f'{c_name}'
        else:
            das = f'{self._sets[0]}'

        if self._quantile is not None and metric_name == 'quantile':
            metric_full_name = f'{metric_name} {self._quantile}'
        else:
            metric_full_name = metric_name

        if self._transform == 'log':
            title = f'{self._varname}: log10({metric_full_name})'
        else:
            title = f'{self._varname}: {metric_full_name}'

        if self._plot_type == 'spatial':
            title = f'{das}: {title}'

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
        self,
        ax,
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

    def spatial_plot(self, da_sets, titles):
        nrows = int((da_sets.sets.size + 1) / 2)
        if len(da_sets) == 1:
            ncols = 1
        else:
            ncols = 2
        if self._metric == 'zscore':
            ncols = 1
            nrows = len(da_sets)

        lat_sets = [da_sets[i]['lat'] for i in range(da_sets.sets.size)]
        cy_datas = {}
        lon_sets = {}
        for i in range(da_sets.sets.size):
            cy_datas[i], lon_sets[i] = add_cyclic_point(da_sets[i], coord=da_sets[i]['lon'])

        fig = plt.figure(dpi=300, figsize=(9, 2.5 * nrows))

        mymap = copy.copy(mpl.cm.get_cmap(f'{self._color}'))
        mymap.set_under(color='black')
        mymap.set_over(color='white')
        mymap.set_bad(alpha=0)

        axs = {}
        psets = {}
        for i in range(da_sets.sets.size):
            axs[i] = plt.subplot(
                nrows, ncols, i + 1, projection=ccrs.Robinson(central_longitude=0.0)
            )

            axs[i].set_facecolor('#39ff14')

            no_inf_data_set = np.nan_to_num(cy_datas[i], nan=np.nan)
            color_min = min(
                [
                    np.min(da_sets[i].where(da_sets[i] != -inf)).values.min()
                    for i in range(da_sets.sets.size)
                ]
            )
            color_max = max(
                [
                    np.max(da_sets[i].where(da_sets[i] != inf)).values.max()
                    for i in range(da_sets.sets.size)
                ]
            )
            psets[i] = axs[i].pcolormesh(
                lon_sets[i],
                lat_sets[i],
                no_inf_data_set,
                transform=ccrs.PlateCarree(),
                cmap=mymap,
                vmin=color_min,
                vmax=color_max,
            )
            axs[i].set_global()

            # if we want to get the ssim
            if self._calc_ssim:
                axs[i].axis('off')
                plt.margins(0, 0)
                extent1 = axs[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                axs[i].imshow
                plt.savefig(f'tmp_ssim{i+1}', bbox_inches=extent1, transparent=True, pad_inches=0)
                axs[i].axis('on')

            axs[i].coastlines()
            axs[i].set_title(titles[i])

        # add colorbar
        fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.95)

        cbs = []
        if not all([np.isnan(cy_datas[i]).all() for i in range(len(cy_datas))]):
            cax = fig.add_axes([0.1, 0, 0.8, 0.05])
            if any([np.isinf(cy_datas[i]).any() for i in range(len(cy_datas))]):
                for i in range(len(psets)):
                    cbs.append(
                        fig.colorbar(
                            psets[i], cax=cax, orientation='horizontal', shrink=0.95, extend='both'
                        )
                    )
                    cbs[i].ax.set_title(f'{da_sets[i].units}')
            else:
                for i in range(len(psets)):
                    cbs.append(
                        fig.colorbar(psets[i], cax=cax, orientation='horizontal', shrink=0.95)
                    )
                    cbs[i].ax.set_title(f'{da_sets[i].units}')
                    cbs[i].ax.tick_params(labelsize=8, rotation=30)
            if any([np.isnan(cy_datas[i]).any() for i in range(len(cy_datas))]):
                proxy = [plt.Rectangle((0, 0), 1, 1, fc='#39ff14')]
                plt.legend(proxy, ['NaN'], bbox_to_anchor=(0.565, 4))
        else:
            fig.add_axes([0.1, 0, 0.8, 0.05])
            proxy = [plt.Rectangle((0, 0), 1, 1, fc='#39ff14')]
            plt.legend(proxy, ['NaN'], bbox_to_anchor=(0.565, 4))
            plt.axis('off')

        if self._calc_ssim:
            import os

            import cv2
            from skimage.metrics import structural_similarity as ssim

            for i in range(1, len(da_sets)):
                img1 = cv2.imread('tmp_ssim1.png')
                img2 = cv2.imread(f'tmp_ssim{i+1}.png')
                # print(img1.shape)
                # print(img2.shape)
                ssim_val = ssim(img1, img2, multichannel=True)
                print(f' SSIM 1 & {i+1} = % 5.5f\n' % (ssim_val))
            for i in range(len(da_sets) + 1):
                if os.path.exists(f'tmp_ssim{i}.png'):
                    os.remove(f'tmp_ssim{i}.png')

    def hist_plot(self, plot_data, title):
        fig, axs = mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(plot_data, label=plot_data.sets.data)
        if plot_data.units != '':
            mpl.pyplot.xlabel(f'{self._metric} ({plot_data.units})')
        else:
            mpl.pyplot.xlabel(f'{self._metric}')
        mpl.pyplot.title(f'time-series histogram: {title[0]}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)

    def periodogram_plot(self, plot_data, title):
        plt.figure()
        for j in range(plot_data.sets.size):
            dat = xrft.dft(
                xr.DataArray(plot_data[j].data - plot_data[j].data.mean()).chunk(
                    (plot_data[j].data - plot_data[j].data.mean()).size
                )
            )
            i = (np.multiply(dat, np.conj(dat)) / dat.size).real
            i = np.log10(i[2 : int(dat.size / 2) + 1])
            freqs = np.array(range(1, int(dat.size / 2))) / dat.size

            mpl.pyplot.plot(freqs, i, label=plot_data[j].sets.data)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
        mpl.pyplot.title(f'periodogram: {title[0]}')
        mpl.pyplot.ylabel('Spectrum')
        mpl.pyplot.xlabel('Frequency')

    def time_series_plot(
        self,
        da_sets,
        titles,
    ):
        """
        time series plot
        """
        group_string = 'time.year'
        xlabel = 'date'
        tick_interval = int(da_sets.size / da_sets.sets.size / 5) + 1
        if da_sets.size / da_sets.sets.size == 1:
            tick_interval = 1
        if self._group_by == 'time.dayofyear':

            group_string = 'dayofyear'
            xlabel = 'Day of Year'
        elif self._group_by == 'time.month':
            group_string = 'month'
            xlabel = 'Month'
            tick_interval = 1
        elif self._group_by == 'time.year':
            group_string = 'year'
            xlabel = 'Year'
        elif self._group_by == 'time.day':
            group_string = 'day'
            xlabel = 'Day'

        if self._metric_type == 'diff':
            if da_sets.units != '':
                ylabel = f'{self._metric} ({da_sets.units}) diff'
            else:
                ylabel = f'{self._metric} diff'
        elif self._metric_type == 'ratio':
            ylabel = f'ratio {self._metric}'
        else:
            if da_sets.units != '':
                ylabel = f'{self._metric} ({da_sets.units})'
            else:
                ylabel = f'{self._metric} diff'

        if self._transform == 'log':
            plot_ylabel = f'log10({ylabel})'
        else:
            plot_ylabel = ylabel

        mpl.style.use('default')

        plt.figure()
        for i in range(da_sets.sets.size):
            if self._group_by is not None:
                plt.plot(
                    da_sets[i][group_string].data,
                    da_sets[i],
                    f'C{i}',
                    label=f'{da_sets.sets.data[i]}',
                )
                ax = plt.gca()
            else:
                # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
                # plt.gca().xaxis.set_major_locator(mdates.DayLocator())
                # dtindex = da_sets[i].indexes['time'].to_datetimeindex()
                # da_sets[i]['time'] = dtindex

                # mpl.pyplot.plot_date(
                #    da_sets[i].time.data, da_sets[i], f'C{i}', label=f'{da_sets.sets.data[i]}'
                # )
                dtindex = da_sets[i].indexes['time']
                c_d_time = [nc_time_axis.CalendarDateTime(item, '365_day') for item in dtindex]
                mpl.pyplot.plot(c_d_time, da_sets[i], f'C{i}', label=f'{da_sets.sets.data[i]}')
                ax = plt.gca()

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
        mpl.pyplot.ylabel(plot_ylabel)
        mpl.pyplot.yscale(self._scale)
        self._label_offset(ax)
        mpl.pyplot.xlabel(xlabel)

        if self._group_by is not None:
            mpl.pyplot.xticks(
                np.arange(min(da_sets[group_string]), max(da_sets[group_string]) + 1, tick_interval)
            )
            if self._group_by == 'time.month':
                int_labels = plt.xticks()[0]
                month_labels = [
                    calendar.month_name[i] for i in int_labels if calendar.month_name[i] != ''
                ]
                unique_month_labels = list(dict.fromkeys(month_labels))
                plt.gca().set_xticklabels(unique_month_labels)
                plt.xticks(rotation=45)
        # else:
        #    mpl.pyplot.xticks(
        #        pd.date_range(
        #            np.datetime64(da_sets['time'].data[0]),
        #            np.datetime64(da_sets['time'].data[-1]),
        #            periods=int(da_sets['time'].size / tick_interval) + 1,
        #        )
        #    )

        mpl.pyplot.title(titles[0])

    def get_metric_label(self, metric, data, weights=None):
        # Get special metric names
        if metric == 'zscore':
            zscore_cutoff = lm.DatasetMetrics((data), ['time']).get_single_metric('zscore_cutoff')
            percent_sig = lm.DatasetMetrics((data), ['time']).get_single_metric(
                'zscore_percent_significant'
            )
            metric_name = f'{metric}: cutoff {zscore_cutoff[0]:.2f}, % sig: {percent_sig:.2f}'
        elif metric == 'mean' and self._plot_type == 'spatial' and self._metric_type == 'raw':
            o_wt_mean = np.average(
                np.average(
                    lm.DatasetMetrics(data, ['time']).get_metric(metric),
                    axis=0,
                    weights=weights,
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
    sets,
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
    quantile=None,
    start=None,
    end=None,
    calc_ssim=False,
):
    """
    Plots the data given an xarray dataset


    Parameters
    ==========
    ds : xarray.Dataset
        The input dataset
    varname : str
        The name of the variable to be plotted
    metric : str
        The name of the metric to be plotted (must match a property name in the DatasetMetrics
        class in ldcpy.plot, for more information about the available metrics see ldcpy.DatasetMetrics)
        Accept values include:

            - ns_con_var
            - ew_con_var
            - mean
            - std
            - variance
            - prob_positive
            - prob_negative
            - odds_positive
            - zscore
            - mean_abs
            - mean_squared
            - rms
            - sum
            - sum_squared
            - corr_lag1
            - quantile
            - lag1
    sets : list <str>
        The labels of the dataset to gather metrics from
    group_by : str
        how to group the data in time series plots.
        Valid groupings:

            - time.day
            - time.dayofyear
            - time.month
            - time.year
    scale : str, optional
        time-series y-axis plot transformation. (default "linear")
        Valid options:

            - linear
            - log
    metric_type : str, optional
        The type of operation to be performed on the metrics. (default 'raw')
        Valid options:

            - raw: the unaltered metric values
            - diff: the difference between the metric values in the first set and every other set
            - ratio: the ratio of the metric values in (2nd, 3rd, 4th... sets/1st set)
            - metric_of_diff: the metric value computed on the difference between the first set and every other set
    plot_type : str , optional
        The type of plot to be created. (default 'spatial')
        Valid options:

            - spatial: a plot of the world with values at each lat and lon point (takes the mean across the time dimension)
            - time-series: A time-series plot of the data (computed by taking the mean across the lat and lon dimensions)
            - histogram: A histogram of the time-series data
    transform : str, optional
        data transformation. (default 'none')
        Valid options:

            - none
            - log
    subset : str, optional
        subset of the data to gather metrics on (default None).
        Valid options:

            - first5: the first 5 days of data
            - winter: data from the months December, January, February
            - spring: data from the months March, April, May
            - summer: data from the months June, July, August
            - autumn: data from the months September, October, November
    lat : float, optional
        The latitude of the data to gather metrics on (default None).
    lon : float , optional
        The longitude of the data to gather metrics on (default None).
    lev : float, optional
        The level of the data to gather metrics on (used if plotting from a 3d data set),
        (default 0).
    color : str, optional
        The color scheme for spatial plots, (default 'coolwarm').
        see https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        for more options
    standardized_err : bool, optional
        Whether or not to standardize the error in a plot of metric_type="diff",
        (default False).
    quantile : float, optional
        A value between 0 and 1 required if metric="quantile", corresponding to the desired quantile to gather,
        (default 0.5).
    start : int, optional
        A value between 0 and the number of time slices indicating the start time of a subset,
        (default None).
    end : int, optional
        A value between 0 and the number of time slices indicating the end time of a subset,
        (default None)
    calc_ssim : bool, optional
        Whether or not to calculate the ssim (structural similarity index) between two plots
        (only applies to plot_type = 'spatial'), (default False).

    Returns
    =======
    out : None
    """

    mp = MetricsPlot(
        ds,
        varname,
        metric,
        sets,
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
        calc_ssim,
    )

    mp.verify_plot_parameters()

    # Subset data
    dss = []
    if 'collection' in ds[varname].dims:
        if sets is not None:
            for set in sets:
                dss.append(ds[varname].sel(collection=set))
    else:
        dss.append(ds[varname])

    subsets = []
    if sets is not None:
        for i in range(len(sets)):
            subsets.append(lu.subset_data(dss[i], subset, lat, lon, lev, start, end))

    # Acquire raw metric values
    datas = []
    if metric_type in ['metric_of_diff']:
        if subsets is not None:
            for i in range(1, len(subsets)):
                datas.append(subsets[0] - subsets[i])
                datas[i - 1].attrs = subsets[0].attrs
    else:
        if subsets is not None:
            for i in range(len(subsets)):
                datas.append(subsets[i])

    raw_metrics = []
    for d in datas:
        raw_metrics.append(mp.get_metrics(d))

    # Get metric names/values for plot title
    # if metric == 'zscore':
    metric_names = []
    for i in range(len(datas)):
        if ds.variables.mapping.get('gw') is not None:
            metric_names.append(mp.get_metric_label(metric, datas[i], ds['gw'].values))
        else:
            metric_names.append(mp.get_metric_label(metric, datas[i]))

    # Get plot data and title
    if lat is not None and lon is not None:
        mp.title_lat = subsets[0]['lat'].data[0]
        mp.title_lon = subsets[0]['lon'].data[0] - 180
    else:
        mp.title_lat = lat
        mp.title_lon = lon

    plot_datas = []
    set_names = []
    if metric_type in ['diff', 'ratio']:
        for i in range(1, len(raw_metrics)):
            plot_datas.append(mp.get_plot_data(raw_metrics[0], raw_metrics[i]))
            set_names.append(f'{sets[0]} & {sets[i]}')
    else:
        for i in range(len(raw_metrics)):
            plot_datas.append(mp.get_plot_data(raw_metrics[i]))
            if metric_type in ['metric_of_diff']:
                set_names.append(f'{sets[0]} & {sets[i]}')
            else:
                set_names.append(f'{sets[i]}')

    plot_dataset = xr.concat(plot_datas, 'sets')
    plot_dataset = plot_dataset.assign_coords({'sets': set_names})

    titles = []

    if metric_type in ['ratio', 'diff']:
        for i in range(1, len(metric_names)):
            titles.append(mp.get_title(metric_names[i], f'{sets[0]} & {sets[i]}'))
    elif metric_type in ['metric_of_diff']:
        for i in range(len(metric_names)):
            titles.append(mp.get_title(metric_names[i], f'{sets[0]} & {sets[i+1]}'))
    else:
        for i in range(len(metric_names)):
            titles.append(mp.get_title(metric_names[i], sets[i]))

    # Call plot functions
    if plot_type == 'spatial':
        mp.spatial_plot(plot_dataset, titles)
    elif plot_type == 'time_series':
        mp.time_series_plot(plot_dataset, titles)
    elif plot_type == 'histogram':
        mp.hist_plot(plot_dataset, titles)
    elif plot_type == 'periodogram':
        mp.periodogram_plot(plot_dataset, titles)
