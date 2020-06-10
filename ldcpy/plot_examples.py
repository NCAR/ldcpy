import matplotlib as mpl
import numpy as np
import scipy.stats as ss
import xarray as xr
import xrft as xrft

import ldcpy.metrics as lm
import ldcpy.plot as lp


def _get_raw_data(da1, metric, plot_type, metric_type, group_by, da2=None):
    if plot_type == 'spatial':
        if metric_type == 'diff_metric':
            metrics_da1 = lm.AggregateMetrics(da1 - da2, ['time'], group_by)
        else:
            metrics_da1 = lm.AggregateMetrics(da1, ['time'], group_by)
    elif plot_type == 'time_series' or plot_type == 'periodogram' or plot_type == 'histogram':
        if metric_type == 'diff_metric':
            metrics_da1 = lm.AggregateMetrics(da1 - da2, ['lat', 'lon'], group_by)
        else:
            metrics_da1 = lm.AggregateMetrics(da1, ['lat', 'lon'], group_by)
    else:
        raise ValueError(f'plot type {plot_type} not supported')

    raw_data = metrics_da1.get_metric(metric)

    return raw_data


def _get_plot_data(raw_data_1, metric_type, transform, raw_data_2=None):
    if metric_type == 'diff':
        plot_data = raw_data_1 - raw_data_2
    elif metric_type == 'ratio':
        plot_data = raw_data_2 / raw_data_1
    elif metric_type == 'raw' or metric_type == 'diff_metric':
        plot_data = raw_data_1
    else:
        raise ValueError(f'metric_type {metric_type} not supported')

    if transform == 'linear':
        pass
    elif transform == 'log':
        plot_data = xr.ufuncs.log10(plot_data)
    else:
        raise ValueError(f'metric transformation {transform} not supported')

    return plot_data


def _get_title(metric_name, ens_o, varname, metric_type, transform, group_by=None, ens_r=None):
    if transform == 'log':
        title = f'{ens_o}: {varname}: log10({metric_name}) {metric_type}'
    else:
        title = f'{ens_o}: {varname}: {metric_name} {metric_type}'

    if group_by is not None:
        title = f'{title} by {group_by}'

    return title


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
    transform='linear',
    subset=None,
    lat=None,
    lon=None,
    color='coolwarm',
    standardized_err=False,
):
    # Subset data
    if subset is not None:
        data_o = _subset_data(ds[varname].sel(ensemble=ens_o), subset, lat, lon)
        if ens_r is not None:
            data_r = _subset_data(ds[varname].sel(ensemble=ens_r), subset, lat, lon)
    else:
        data_o = ds[varname].sel(ensemble=ens_o)
        if ens_r is not None:
            data_r = ds[varname].sel(ensemble=ens_r)

    # Acquire raw data
    raw_data_o = _get_raw_data(data_o, metric, 'spatial', metric_type, group_by, da2=data_r)
    if ens_r is not None:
        raw_data_r = _get_raw_data(data_r, metric, 'spatial', 'raw', group_by)
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
    if plot_type == 'spatial_comparison':
        plot_data_o = _get_plot_data(raw_data_o, metric_type, transform)
        plot_data_r = _get_plot_data(raw_data_r, metric_type, transform)
        title_o = _get_title(metric_name_o, ens_o, varname, metric_type, transform, group_by)
        title_r = _get_title(metric_name_r, ens_r, varname, metric_type, transform, group_by)
    else:
        plot_data = _get_plot_data(raw_data_o, metric_type, transform, raw_data_2=raw_data_r)
        title = _get_title(metric_name, ens_o, varname, metric_type, transform, group_by)

    # Call plot functions
    if plot_type == 'spatial_comparison':
        lp.spatial_comparison_plot(plot_data_o, title_o, plot_data_r, title_r, color)
    if plot_type == 'spatial':
        lp.spatial_plot(plot_data, title, color)


def compare_plot(
    ds,
    varname,
    method_str,
    ens_o,
    ens_r,
    metric='mean',
    color='cmo.thermal',
    transform='none',
    grouping='time.dayofyear',
    subset=None,
    lat=None,
    lon=None,
):

    if subset is not None:
        data_o = _subset_data(ds[varname].sel(ensemble=ens_o), subset, lat, lon)
        data_r = _subset_data(ds[varname].sel(ensemble=ens_r), subset, lat, lon)
    else:
        data_o = ds[varname].sel(ensemble=ens_o)
        data_r = ds[varname].sel(ensemble=ens_r)

    metrics_o = lm.AggregateMetrics(data_o, ['time'], grouping).get_metric(metric)
    metrics_r = lm.AggregateMetrics(data_r, ['time'], grouping).get_metric(metric)
    title_1 = f'{metric}'
    title_2 = f'{metric}'
    if metric == 'mean':
        gw = ds['gw'].values
        o_wt_mean = np.average(np.average(metrics_o, axis=0, weights=gw))
        r_wt_mean = np.average(np.average(metrics_r, axis=0, weights=gw))
        title_1 = f'mean = {o_wt_mean:.2f}'
        title_2 = f'mean = {r_wt_mean:.2f}'

    if transform == 'log':
        metrics_o = xr.ufuncs.log10(metrics_o)
        metrics_r = xr.ufuncs.log10(metrics_r)
        title_1 = f'log10({metric})'
        title_2 = f'log10({metric})'

    lp.spatial_comparison_plot(
        metrics_o, metrics_r, varname, method_str, title_1, title_2, color,
    )


def single_plot(
    ds,
    varname,
    method_str,
    ens_o,
    ens_r=None,
    metric='mean',
    color='coolwarm',
    metric_type='diff',
    transform='none',
    grouping='dayofyear',
    subset=None,
    lat=None,
    lon=None,
):

    if subset is not None:
        data_o = _subset_data(ds[varname].sel(ensemble=ens_o), subset, lat, lon)
        data_r = _subset_data(ds[varname].sel(ensemble=ens_r), subset, lat, lon)
    else:
        data_o = ds[varname].sel(ensemble=ens_o)
        data_r = ds[varname].sel(ensemble=ens_r)

    metric_o = lm.AggregateMetrics(data_o, ['time']).get_metric(metric)
    colormap = 'coolwarm'

    if metric_type == 'diff':
        metric_r = lm.AggregateMetrics(data_r, ['time']).get_metric(metric)
        title = f'{varname} ({method_str}): {metric} diff'
        plot_data = metric_o - metric_r
    elif metric_type == 'ratio':
        metric_r = lm.AggregateMetrics(data_r, ['time']).get_metric(metric)
        title = f'{varname} ({method_str}): {metric} ratio'
        plot_data = metric_r / metric_o
    elif metric_type == 'diff_metric':
        metric_e = lm.AggregateMetrics(data_o - data_r, ['time']).get_metric(metric)
        title = f'{metric}'
        plot_data = metric_e
    elif metric_type == 'maeday':
        # waiting on xarray 0.15.2
        # max_day_e = lm.AggregateMetrics(data_o - data_r, ['time'], "time.dayofyear").get_metric("mae_max")
        metric_e = np.abs((data_o - data_r).groupby('time.dayofyear').mean(dim='time'))
        max_day_e = metric_e.argmax(dim=grouping)
        title = f'{metric}'
        plot_data = max_day_e
        colormap = 'twilight'
    else:
        title = f'{metric}'
        plot_data = metric_o

    if transform == 'log':
        plot_data = xr.ufuncs.log10(plot_data)
        title = f'{varname} ({method_str}): {transform}({metric}) ratio'

    if metric == 'zscore':
        zscore_cutoff = lm.OverallMetrics((data_o - data_r), ['time']).get_overall_metric(
            'zscore_cutoff'
        )
        percent_sig = lm.OverallMetrics((data_o - data_r), ['time']).get_overall_metric(
            'zscore_percent_significant'
        )
        name = f'{metric}: cutoff {zscore_cutoff[0]:.2f}, % sig: {percent_sig:.2f}'
        title = f'{varname} ({method_str}): {name}'

    lp.spatial_plot(title, plot_data, colormap)


def time_series_plot(
    ds,
    varname,
    method_str,
    ens_o,
    ens_r=None,
    grouping='None',
    scale='linear',
    metric='mean',
    metric_type='n',
    plot_type='timeseries',
    transform='none',
    subset=None,
    lat=None,
    lon=None,
    standardized_err=False,
):
    """
    time series plot
    """
    group_string = 'time.year'
    if grouping == 'dayofyear':
        group_string = 'time.dayofyear'
        xlabel = 'Day of Year'
        tick_interval = 20
    elif grouping == 'month':
        group_string = 'time.month'
        xlabel = 'Month'
        tick_interval = 1
    elif grouping == 'year':
        group_string = 'time.year'
        xlabel = 'Year'
        tick_interval = 1
    elif grouping == 'day':
        group_string = 'time.day'
        xlabel = 'Day'
        tick_interval = 20

    if subset is not None:
        data_o = _subset_data(ds[varname].sel(ensemble=ens_o), subset, lat, lon)
        data_r = _subset_data(ds[varname].sel(ensemble=ens_r), subset, lat, lon)
    else:
        data_o = ds[varname].sel(ensemble=ens_o)
        data_r = ds[varname].sel(ensemble=ens_r)

    time_series_metrics_o = lm.AggregateMetrics(data_o, ['lat', 'lon'])

    if metric_type == 'diff':
        time_series_metrics_r = lm.AggregateMetrics(data_r, ['lat', 'lon'])
        error_data = time_series_metrics_o.get_metric(metric) - time_series_metrics_r.get_metric(
            metric
        )
        if standardized_err is True:
            if error_data.std(dim='time') != 0:
                error_data = (error_data - error_data.mean(dim='time')) / error_data.std(dim='time')
            else:
                raise ValueError(
                    'Standard deviation of error data is 0. Cannot standardize errors.'
                )
        grouped_metric = (error_data).groupby(group_string)
        ylabel = f'{metric} error'
    elif metric_type == 'ratio':
        time_series_metrics_r = lm.AggregateMetrics(data_r, ['lat', 'lon'])
        grouped_metric = (
            time_series_metrics_r.get_metric(metric) / time_series_metrics_o.get_metric(metric)
        ).groupby(group_string)
        ylabel = f'ratio {ens_r}/{ens_o} {metric}'
    else:
        grouped_metric = (time_series_metrics_o.get_metric(metric)).groupby(group_string)
        ylabel = f'{metric}'

    plot_data = grouped_metric.mean(dim='time')

    if transform == 'log':
        plot_data = xr.ufuncs.log10(plot_data)

    title = f'{varname} ({method_str}): {metric} by {xlabel}'

    if transform == 'none':
        plot_ylabel = ylabel
    elif transform == 'log':
        plot_ylabel = f'log10({ylabel})'

    mpl.pyplot.plot(plot_data[grouping].data, plot_data)

    mpl.pyplot.ylabel(plot_ylabel)
    mpl.pyplot.yscale(scale)
    mpl.pyplot.xlabel(xlabel)
    mpl.pyplot.xticks(
        np.arange(min(plot_data[grouping]), max(plot_data[grouping]) + 1, tick_interval)
    )
    mpl.pyplot.title(title)

    if plot_type == 'hist':
        fig, axs = mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(plot_data)

    if plot_type == 'periodogram':
        dat = xrft.dft(plot_data - plot_data.mean())
        i = (np.multiply(dat, np.conj(dat)) / dat.size).real
        i = xr.ufuncs.log10(i[2 : int(dat.size / 2) + 1])
        freqs = np.array(range(1, int(dat.size / 2))) / dat.size

        mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
        mpl.pyplot.plot(freqs, i)


def _subset_data(ds, cond, lat=None, lon=None):
    if cond == 'winter':
        ds_subset = ds.where(ds.time.dt.season == 'DJF', drop=True)
    elif cond == 'first50':
        ds_subset = ds.isel(time=slice(None, 50))

    if lat is not None:
        ds_subset = ds_subset.sel(lat=lat, method='nearest')
        ds_subset = ds_subset.expand_dims('lat')

    if lon is not None:
        ds_subset = ds_subset.sel(lon=lon, method='nearest')
        ds_subset = ds_subset.expand_dims('lon')

    return ds_subset
