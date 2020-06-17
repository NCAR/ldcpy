import math
import sys

import cartopy.crs as ccrs
import cmocean
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import xrft
from cartopy.util import add_cyclic_point
from numpy import inf

import ldcpy.metrics as lm


def _subset_data(ds, cond, lat=None, lon=None, lev=1):

    ds_subset = ds

    if cond == 'winter':
        ds_subset = ds_subset.where(ds.time.dt.season == 'DJF', drop=True)
    elif cond == 'first50':
        ds_subset = ds_subset.isel(time=slice(None, 50))

    if 'lev' in ds_subset.dims:
        ds_subset = ds_subset.sel(lev=0, method='nearest')

    if lat is not None:
        ds_subset = ds_subset.sel(lat=lat, method='nearest')
        ds_subset = ds_subset.expand_dims('lat')

    if lon is not None:
        ds_subset = ds_subset.sel(lon=lon + 180, method='nearest')
        ds_subset = ds_subset.expand_dims('lon')

    return ds_subset


def _get_raw_data(da1, metric, plot_type, metric_type, group_by, mae_group_by, da2=None):
    if plot_type == 'spatial':
        if group_by is not None and metric != 'mae_max':
            raise ValueError(f'cannot group by time in a spatial plot of {metric}.')
        if metric_type == 'diff_metric':
            metrics_da1 = lm.AggregateMetrics(da1 - da2, ['time'], mae_group_by)
        else:
            metrics_da1 = lm.AggregateMetrics(da1, ['time'], mae_group_by)
    elif plot_type == 'time_series' or plot_type == 'periodogram' or plot_type == 'histogram':
        if metric_type == 'diff_metric':
            metrics_da1 = lm.AggregateMetrics(da1 - da2, ['lat', 'lon'], group_by)
        else:
            metrics_da1 = lm.AggregateMetrics(da1, ['lat', 'lon'], group_by)
    else:
        raise ValueError(f'plot type {plot_type} not supported')

    raw_data = metrics_da1.get_metric(metric)

    return raw_data


def _get_plot_data(
    raw_data_1, metric_type, transform, grouping, raw_data_2=None, standardized_err=False,
):
    if metric_type == 'diff':
        plot_data = raw_data_1 - raw_data_2
    elif metric_type == 'ratio':
        plot_data = raw_data_2 / raw_data_1
    elif metric_type == 'raw' or metric_type == 'diff_metric':
        plot_data = raw_data_1
    else:
        raise ValueError(f'metric_type {metric_type} not supported')

    if metric_type == 'diff' and standardized_err is True:
        if plot_data.std(dim='time') != 0:
            plot_data = (plot_data - plot_data.mean(dim='time')) / plot_data.std(dim='time')
        else:
            raise ValueError('Standard deviation of error data is 0. Cannot standardize errors.')

    if grouping is not None:
        plot_data = plot_data.groupby(grouping).mean(dim='time')

    if transform == 'linear':
        pass
    elif transform == 'log':
        plot_data = xr.ufuncs.log10(plot_data)
    else:
        raise ValueError(f'metric transformation {transform} not supported')

    return plot_data


def _get_title(
    metric_name,
    ens_o,
    varname,
    metric_type,
    transform,
    group_by=None,
    ens_r=None,
    lat=None,
    lon=None,
    subset=None,
):
    if ens_r is not None:
        das = f'{ens_o}, {ens_r}'
    else:
        das = f'{ens_o}'

    if transform == 'log':
        title = f'{das}: {varname}: log10({metric_name}) {metric_type}'
    else:
        title = f'{das}: {varname}: {metric_name} {metric_type}'

    if group_by is not None:
        title = f'{title} by {group_by}'

    if lat is not None:
        if lon is not None:
            title = f'{title} at lat={lat:.2f}, lon={lon:.2f}'
        else:
            title = f'{title} at lat={lat:.2f}'
    elif lon is not None:
        title = f'{title} at lat={lon:.2f}'

    if subset is not None:
        title = f'{title} subset:{subset}'

    return title


def _calc_contour_levels(dat_1, dat_2=None, nlevs=None):
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
    levels = minval + np.arange(nlevs + 1) * (maxval - minval) / nlevs
    # print('Min value: {}\nMax value: {}'.format(minval, maxval))
    return levels


def spatial_comparison_plot(da_o, title_o, da_r, title_r, color='cmo.thermal'):
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

    mymap = plt.get_cmap(f'{color}')
    mymap.set_under(color='black')
    mymap.set_over(color='white')
    mymap.set_bad(alpha=0)

    # both plots use same contour levels
    levels = _calc_contour_levels(da_o, da_r, nlevs=24)

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


def spatial_plot(da, title, color='coolwarm'):
    """
    visualize the mean error
    want to be able to input multiple?
    """

    levels = _calc_contour_levels(da, nlevs=24)

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

    mymap = plt.get_cmap(color)
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
        if np.isfinite(da).all() and np.isfinite(da).all():
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


def hist_plot(plot_data, title, color='red'):
    fig, axs = mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(plot_data)


def periodogram_plot(plot_data, title, color='red'):
    dat = xrft.dft(plot_data - plot_data.mean())
    i = (np.multiply(dat, np.conj(dat)) / dat.size).real
    i = xr.ufuncs.log10(i[2 : int(dat.size / 2) + 1])
    freqs = np.array(range(1, int(dat.size / 2))) / dat.size

    mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
    mpl.pyplot.plot(freqs, i)


def time_series_plot(
    da,
    title,
    grouping=None,
    scale='linear',
    metric='mean',
    metric_type='n',
    plot_type='timeseries',
    transform='none',
):
    """
    time series plot
    """
    group_string = 'time.year'
    xlabel = 'date'
    tick_interval = 28
    if grouping == 'time.dayofyear':
        group_string = 'dayofyear'
        xlabel = 'Day of Year'
        tick_interval = 20
    elif grouping == 'time.month':
        group_string = 'month'
        xlabel = 'Month'
        tick_interval = 1
    elif grouping == 'time.year':
        group_string = 'year'
        xlabel = 'Year'
        tick_interval = 1
    elif grouping == 'time.day':
        group_string = 'day'
        xlabel = 'Day'
        tick_interval = 20

    if metric_type == 'diff':
        ylabel = f'{metric} error'
    elif metric_type == 'ratio':
        ylabel = f'ratio {metric}'
    else:
        ylabel = f'{metric}'

    if transform == 'none':
        plot_ylabel = ylabel
    elif transform == 'log':
        plot_ylabel = f'log10({ylabel})'

    if grouping is not None:
        mpl.pyplot.plot(da[group_string].data, da)
    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        dtindex = da.indexes['time'].to_datetimeindex()
        da['time'] = dtindex
        mpl.pyplot.plot_date(da.time.data, da)

    mpl.pyplot.ylabel(plot_ylabel)
    mpl.pyplot.yscale(scale)
    mpl.pyplot.xlabel(xlabel)

    if grouping is not None:
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


def plot(
    ds,
    varname,
    ens_o,
    metric,
    ens_r=None,
    mae_group_by=None,
    group_by=None,
    scale='linear',
    metric_type='raw',
    plot_type='spatial',
    transform='linear',
    subset=None,
    lat=None,
    lon=None,
    lev=0,
    color='coolwarm',
    standardized_err=False,
):
    """
    Plots the data given an xarray dataset (ds), dataset variable (varname), ensemble (ens_o) and desired metric (metric)
    """
    # Subset data
    data_r = None
    data_o = _subset_data(ds[varname].sel(ensemble=ens_o), subset, lat, lon, lev)
    if ens_r is not None:
        data_r = _subset_data(ds[varname].sel(ensemble=ens_r), subset, lat, lon, lev)

    # Acquire raw data
    if plot_type == 'spatial_comparison':
        raw_data_o = _get_raw_data(
            data_o, metric, 'spatial', metric_type, group_by, mae_group_by, da2=data_r
        )
    else:
        raw_data_o = _get_raw_data(
            data_o, metric, plot_type, metric_type, group_by, mae_group_by, da2=data_r
        )
    if ens_r is not None:
        if plot_type == 'spatial_comparison':
            raw_data_r = _get_raw_data(
                data_r, metric, 'spatial', metric_type, group_by, mae_group_by, da2=data_r
            )
        else:
            raw_data_r = _get_raw_data(data_r, metric, plot_type, 'raw', group_by, mae_group_by)
    else:
        raw_data_r = None

    # Get special metric names
    if metric == 'zscore':
        zscore_cutoff = lm.OverallMetrics((data_o - data_r), ['time']).get_overall_metric(
            'zscore_cutoff'
        )
        percent_sig = lm.OverallMetrics((data_o - data_r), ['time']).get_overall_metric(
            'zscore_percent_significant'
        )
        metric_name = f'{metric}: cutoff {zscore_cutoff[0]:.2f}, % sig: {percent_sig:.2f}'
        metric_name_o = metric_name
        metric_name_r = metric_name
    elif metric == 'mean' and plot_type == 'spatial_comparison':
        gw = ds['gw'].values
        o_wt_mean = np.average(
            np.average(
                lm.AggregateMetrics(data_o, ['time'], group_by).get_metric(metric),
                axis=0,
                weights=gw,
            )
        )
        r_wt_mean = np.average(
            np.average(
                lm.AggregateMetrics(data_r, ['time'], group_by).get_metric(metric),
                axis=0,
                weights=gw,
            )
        )
        metric_name_o = f'{metric} = {o_wt_mean:.2f}'
        metric_name_r = f'{metric} = {r_wt_mean:.2f}'
    else:
        metric_name = metric
        metric_name_o = metric
        metric_name_r = metric

    # Get plot data and title based on plot_type
    if lat is not None:
        title_lat = data_o['lat'].data[0]
        title_lon = data_o['lon'].data[0] - 180
    else:
        title_lat = lat
        title_lon = lon

    if plot_type == 'spatial_comparison':
        plot_data_o = _get_plot_data(raw_data_o, metric_type, transform, group_by)
        plot_data_r = _get_plot_data(raw_data_r, metric_type, transform, group_by)
        title_o = _get_title(
            metric_name_o,
            ens_o,
            varname,
            metric_type,
            transform,
            group_by,
            lat=title_lat,
            lon=title_lon,
            subset=subset,
        )
        title_r = _get_title(
            metric_name_r,
            ens_r,
            varname,
            metric_type,
            transform,
            group_by,
            lat=title_lat,
            lon=title_lon,
            subset=subset,
        )
    else:
        plot_data = _get_plot_data(
            raw_data_o,
            metric_type,
            transform,
            group_by,
            raw_data_2=raw_data_r,
            standardized_err=standardized_err,
        )
        title = _get_title(
            metric_name,
            ens_o,
            varname,
            metric_type,
            transform,
            group_by,
            lat=title_lat,
            lon=title_lon,
            subset=subset,
            ens_r=ens_r,
        )

    # Call plot functions
    if plot_type == 'spatial_comparison':
        spatial_comparison_plot(plot_data_o, title_o, plot_data_r, title_r, color)
    elif plot_type == 'spatial':
        spatial_plot(plot_data, title, color)
    elif plot_type == 'time_series':
        time_series_plot(plot_data, title, group_by, metric_type=metric_type)
    elif plot_type == 'histogram':
        hist_plot(plot_data, title)
    elif plot_type == 'periodogram':
        periodogram_plot(plot_data, title)
