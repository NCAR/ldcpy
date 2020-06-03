import numpy as np
import scipy.stats as ss
import xarray as xr

import ldcpy.metrics as lm
import ldcpy.plot as lp


def compare_con_var(ds, varname, ens_o, ens_r, method_str, nlevs=24, dir='NS'):
    """
    TODO: visualize contrast variance at each grid point for orig and compressed (time-series)
    assuming FV mean
    """
    metrics_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time'])
    metrics_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time'])
    lp.compare_plot(
        xr.ufuncs.log10(metrics_o.get_spatial_metric('ns_con_var_spatial')),
        xr.ufuncs.log10(metrics_r.get_spatial_metric('ns_con_var_spatial')),
        varname,
        method_str,
        f'{dir} con_var',
        f'{dir} con_var',
        'binary_r',
        nlevs,
    )


def compare_mean(ds, varname, ens_o, ens_r, method_str, nlevs=24):
    """
    visualize mean value at each grid point for orig and compressed (time-series)
    assuming FV data and put the weighted mean
    """
    metrics_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time'])
    metrics_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time'])
    mean_data_o = metrics_o.get_spatial_metric('mean_spatial')
    mean_data_r = metrics_r.get_spatial_metric('mean_spatial')

    # weighted mean
    gw = ds['gw'].values
    o_wt_mean = np.average(np.average(mean_data_o, axis=0, weights=gw))
    r_wt_mean = np.average(np.average(mean_data_r, axis=0, weights=gw))
    lp.compare_plot(
        mean_data_o,
        mean_data_r,
        varname,
        method_str,
        f'mean = {o_wt_mean:.2f}',
        f'mean = {r_wt_mean:.2f}',
        'cmo.thermal',
        nlevs,
    )


def compare_std(ds, varname, ens_o, ens_r, method_str, nlevs=24):
    """
    TODO: visualize std dev at each grid point for orig and compressed (time-series)
    assuming FV mean
    """
    metrics_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time'])
    metrics_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time'])

    lp.compare_plot(
        metrics_o.get_spatial_metric('std_spatial'),
        metrics_r.get_spatial_metric('std_spatial'),
        varname,
        method_str,
        'std',
        'std',
        'coolwarm',
        nlevs,
    )


def odds_positive_ratio(ds, varname, ens_o, ens_r, method_str):
    metrics_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time'])
    metrics_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time'])

    log_odds_ratio = xr.ufuncs.log10(
        metrics_r.get_spatial_metric('odds_positive_spatial')
        / metrics_o.get_spatial_metric('odds_positive_spatial')
    )
    lp.plot('log10(odds ratio)', log_odds_ratio, 'PRECT', method_str)


def compare_odds_positive(ds, varname, ens_o, ens_r, method_str):
    metrics_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time'])
    metrics_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time'])
    lp.compare_plot(
        xr.ufuncs.log10(metrics_o.get_spatial_metric('odds_positive_spatial')),
        xr.ufuncs.log10(metrics_r.get_spatial_metric('odds_positive_spatial')),
        'PRECT',
        method_str,
        'log10(odds rain)',
        'log10(odds rain)',
        'coolwarm',
    )


def compare_prob_negative(ds, varname, ens_o, ens_r, method_str):
    metrics_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time'])
    metrics_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time'])
    lp.compare_plot(
        metrics_o.get_spatial_metric('prob_negative_spatial'),
        metrics_r.get_spatial_metric('prob_negative_spatial'),
        'PRECT',
        method_str,
        'P(neg rainfall)',
        'P(neg rainfall)',
        'binary',
    )


def diff_mean(ds, varname, ens_o, ens_r, method_str):
    metrics_d = lm.SpatialMetrics(
        ds[varname].sel(ensemble=ens_o) - ds[varname].sel(ensemble=ens_r), ['time']
    )
    lp.plot('mean error', metrics_d.get_spatial_metric('mean_spatial'), varname, method_str)


def diff_std(ds, varname, ens_o, ens_r, method_str):
    std_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time']).get_spatial_metric(
        'std_spatial'
    )
    std_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time']).get_spatial_metric(
        'std_spatial'
    )
    lp.plot('std error', std_o - std_r, varname, method_str)


def diff_negative_rain(ds, varname, ens_o, ens_r, method_str):
    prob_neg_rain_o = lm.SpatialMetrics(
        ds[varname].sel(ensemble=ens_o), ['time']
    ).get_spatial_metric('prob_negative_spatial')
    prob_neg_rain_r = lm.SpatialMetrics(
        ds[varname].sel(ensemble=ens_r), ['time']
    ).get_spatial_metric('prob_negative_spatial')

    lp.plot('P(neg rain) error', prob_neg_rain_o - prob_neg_rain_r, 'PRECT', method_str)


def diff_con_var(ds, varname, ens_o, ens_r, method_str, dir='NS'):
    con_var_ns_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['time']).get_spatial_metric(
        'ns_con_var_spatial'
    )
    con_var_ns_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['time']).get_spatial_metric(
        'ns_con_var_spatial'
    )
    lp.plot(f'{dir} con_var error', con_var_ns_o - con_var_ns_r, varname, method_str)


def diff_zscore(ds, varname, ens_o, ens_r, method_str):
    z_score_d = lm.SpatialMetrics(
        ds[varname].sel(ensemble=ens_o) - ds[varname].sel(ensemble=ens_r), ['time']
    ).get_spatial_metric('zscore_spatial')

    pvals = 2 * (1 - ss.norm.cdf(np.abs(z_score_d)))
    sorted_pvals = np.sort(pvals).flatten()
    fdr_zscore = 0.01
    p = np.argwhere(sorted_pvals <= fdr_zscore * np.arange(1, pvals.size + 1) / pvals.size)
    pval_cutoff = sorted_pvals[p[len(p) - 1]]
    if not (pval_cutoff.size == 0):
        zscore_cutoff = ss.norm.ppf(1 - pval_cutoff)
        sig_locs = np.argwhere(pvals <= pval_cutoff)
        percent_sig = 100 * np.size(sig_locs, 0) / pvals.size
    else:
        zscore_cutoff = 'na'
        percent_sig = 0

    title = f'z-score error: cutoff {zscore_cutoff[0]:.2f}, % sig: {percent_sig:.2f}'

    lp.plot(title, z_score_d, varname, method_str)


def time_series_diff(
    ds,
    varname,
    ens_o,
    ens_r,
    method_str,
    resolution='dayofyear',
    plot_type='normal',
    data='mean_spatial',
):
    group_string = 'time.year'
    if resolution == 'dayofyear':
        group_string = 'time.dayofyear'
    elif resolution == 'month':
        group_string = 'time.month'
    elif resolution == 'year':
        group_string = 'time.year'

    time_series_metrics_e = lm.SpatialMetrics(
        ds[varname].sel(ensemble=ens_o) - ds[varname].sel(ensemble=ens_r), ['lat', 'lon']
    )
    grouped_metrics_e = time_series_metrics_e.get_spatial_metric(data).groupby(group_string)
    plot_data = grouped_metrics_e.mean(dim='time')

    lp.time_series_plot(plot_data, varname, method_str, resolution, plot_type)


def time_series_plot(
    ds, varname, ens, method_str, resolution='dayofyear', plot_type='normal', data='mean_spatial'
):
    group_string = 'time.year'
    if resolution == 'dayofyear':
        group_string = 'time.dayofyear'
    elif resolution == 'month':
        group_string = 'time.month'
    elif resolution == 'year':
        group_string = 'time.year'

    time_series_metrics = lm.SpatialMetrics(ds[varname].sel(ensemble=ens), ['lat', 'lon'])
    grouped_metrics = time_series_metrics.get_spatial_metric(data).groupby(group_string)
    plot_data = grouped_metrics.mean(dim='time')

    lp.time_series_plot(plot_data, varname, method_str, resolution, plot_type)


def time_series_ratio(
    ds,
    varname,
    ens_o,
    ens_r,
    method_str,
    resolution='dayofyear',
    plot_type='normal',
    data='mean_spatial',
):
    group_string = 'time.year'
    if resolution == 'dayofyear':
        group_string = 'time.dayofyear'
    elif resolution == 'month':
        group_string = 'time.month'
    elif resolution == 'year':
        group_string = 'time.year'

    time_series_metrics_o = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ['lat', 'lon'])
    grouped_metrics_o = time_series_metrics_o.get_spatial_metric(data).groupby(group_string)

    time_series_metrics_r = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_r), ['lat', 'lon'])
    grouped_metrics_r = time_series_metrics_r.get_spatial_metric(data).groupby(group_string)

    plot_data = grouped_metrics_r.mean(dim='time') / grouped_metrics_o.mean(dim='time')

    lp.time_series_plot(plot_data, varname, method_str, resolution, plot_type)
