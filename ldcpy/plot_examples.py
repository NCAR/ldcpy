import matplotlib as mpl
import numpy as np
import scipy.stats as ss
import xarray as xr

import ldcpy.metrics as lm
import ldcpy.plot as lp


def compare_plot(
    ds, varname, method_str, ens_o, ens_r, data='mean', color='cmo.thermal', transform='none'
):
    metrics_o = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_o), ['time']).get_metric(data)
    metrics_r = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_r), ['time']).get_metric(data)
    title_1 = f'{data}'
    title_2 = f'{data}'
    if data == 'mean':
        gw = ds['gw'].values
        o_wt_mean = np.average(np.average(metrics_o, axis=0, weights=gw))
        r_wt_mean = np.average(np.average(metrics_r, axis=0, weights=gw))
        title_1 = f'mean = {o_wt_mean:.2f}'
        title_2 = f'mean = {r_wt_mean:.2f}'

    if transform == 'log':
        metrics_o = xr.ufuncs.log10(metrics_o)
        metrics_r = xr.ufuncs.log10(metrics_r)
        title_1 = f'log10({data})'
        title_2 = f'log10({data})'

    lp.spatial_comparison_plot(
        metrics_o, metrics_r, varname, method_str, title_1, title_2, color,
    )


def single_plot(
    ds, varname, method_str, ens_o, ens_r=None, data='mean', plot_type='diff', transform='none'
):
    metric_o = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_o), ['time']).get_metric(data)

    if plot_type == 'diff':
        metric_r = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_r), ['time']).get_metric(data)
        title = f'{varname} ({method_str}): {data} diff'
        plot_data = metric_o - metric_r
    elif plot_type == 'ratio':
        metric_r = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_r), ['time']).get_metric(data)
        title = f'{varname} ({method_str}): {data} ratio'
        plot_data = metric_r / metric_o
    elif plot_type == 'diff_metric':
        metric_e = lm.AggregateMetrics(
            ds[varname].sel(ensemble=ens_o) - ds[varname].sel(ensemble=ens_r), ['time']
        ).get_metric(data)
        title = f'{data}'
        plot_data = metric_e
    else:
        title = f'{data}'
        plot_data = metric_o

    if transform == 'log':
        plot_data = xr.ufuncs.log10(plot_data)
        title = f'{varname} ({method_str}): {transform}({data}) ratio'

    if data == 'zscore':
        zscore_cutoff = lm.OverallMetrics(
            (ds[varname].sel(ensemble=ens_o) - ds[varname].sel(ensemble=ens_r)), ['time']
        ).get_overall_metric('zscore_cutoff')
        percent_sig = lm.OverallMetrics(
            (ds[varname].sel(ensemble=ens_o) - ds[varname].sel(ensemble=ens_r)), ['time']
        ).get_overall_metric('zscore_percent_significant')
        name = f'{data}: cutoff {zscore_cutoff[0]:.2f}, % sig: {percent_sig:.2f}'
        title = f'{varname} ({method_str}): {name}'

    lp.spatial_plot(title, plot_data)


def time_series_plot(
    ds,
    varname,
    method_str,
    ens_o,
    ens_r=None,
    res='dayofyear',
    scale='linear',
    data='mean',
    plot_type='n',
    transform='none',
):
    """
    time series plot
    """
    group_string = 'time.year'
    if res == 'dayofyear':
        group_string = 'time.dayofyear'
        xlabel = 'Day of Year'
        tick_interval = 20
    elif res == 'month':
        group_string = 'time.month'
        xlabel = 'Month'
        tick_interval = 1
    elif res == 'year':
        group_string = 'time.year'
        xlabel = 'Year'
        tick_interval = 1

    time_series_metrics_o = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_o), ['lat', 'lon'])
    if plot_type == 'diff':
        time_series_metrics_r = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_r), ['lat', 'lon'])
        grouped_metric = (
            time_series_metrics_o.get_metric(data) - time_series_metrics_r.get_metric(data)
        ).groupby(group_string)
        ylabel = f'{data} error'
    elif plot_type == 'ratio':
        time_series_metrics_r = lm.AggregateMetrics(ds[varname].sel(ensemble=ens_r), ['lat', 'lon'])
        grouped_metric = (
            time_series_metrics_r.get_metric(data) / time_series_metrics_o.get_metric(data)
        ).groupby(group_string)
        ylabel = f'ratio {ens_r}/{ens_o} {data}'
    else:
        grouped_metric = (time_series_metrics_o.get_metric(data)).groupby(group_string)
        ylabel = f'{data}'

    plot_data = grouped_metric.mean(dim='time')
    if transform == 'log':
        plot_data = xr.ufuncs.log10(plot_data)

    title = f'{varname} ({method_str}): {data} by {xlabel}'

    if transform == 'none':
        plot_ylabel = ylabel
    elif transform == 'log':
        plot_ylabel = f'log10({ylabel})'

    mpl.pyplot.plot(plot_data[res].data, plot_data)

    mpl.pyplot.ylabel(plot_ylabel)
    mpl.pyplot.yscale(scale)
    mpl.pyplot.xlabel(xlabel)
    mpl.pyplot.xticks(np.arange(min(plot_data[res]), max(plot_data[res]) + 1, tick_interval))
    mpl.pyplot.title(title)
