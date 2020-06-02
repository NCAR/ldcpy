import numpy as np
import xarray as xr

import ldcpy.metrics as lm
import ldcpy.plot as lp


def compare_con_var(ds, varname, ens_o, ens_r, method_str, nlevs=24, dir='NS'):
    """
    TODO: visualize contrast variance at each grid point for orig and compressed (time-series)
    assuming FV mean
    """
    metrics = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ds[varname].sel(ensemble=ens_r))
    lp.compare_plot(
        xr.ufuncs.log10(metrics.get_spatial_metrics_by_name('ns_con_var_orig_spatial')),
        xr.ufuncs.log10(metrics.get_spatial_metrics_by_name('ns_con_var_compressed_spatial')),
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
    metrics = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ds[varname].sel(ensemble=ens_r))
    mean_data_o = metrics.get_spatial_metrics_by_name('mean_orig_spatial')
    mean_data_r = metrics.get_spatial_metrics_by_name('mean_compressed_spatial')

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
    metrics = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ds[varname].sel(ensemble=ens_r))

    lp.compare_plot(
        metrics.get_spatial_metrics_by_name('std_orig_spatial'),
        metrics.get_spatial_metrics_by_name('std_compressed_spatial'),
        varname,
        method_str,
        'std',
        'std',
        'coolwarm',
        nlevs,
    )


def odds_positive_ratio(ds, varname, ens_o, ens_r, method_str):
    metrics = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ds[varname].sel(ensemble=ens_r))

    log_odds_ratio = xr.ufuncs.log10(
        metrics.odds_positive_compressed_spatial / metrics.odds_positive_orig_spatial
    )
    lp.plot_error('log10(odds ratio)', log_odds_ratio, 'PRECT', method_str)


def compare_odds_positive(ds, varname, ens_o, ens_r, method_str):
    metrics = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ds[varname].sel(ensemble=ens_r))
    lp.compare_plot(
        xr.ufuncs.log10(metrics.odds_positive_orig_spatial),
        xr.ufuncs.log10(metrics.odds_positive_compressed_spatial),
        'PRECT',
        method_str,
        'log10(odds rain)',
        'log10(odds rain)',
        'coolwarm',
    )


def compare_prob_neg(ds, varname, ens_o, ens_r, method_str):
    metrics = lm.SpatialMetrics(ds[varname].sel(ensemble=ens_o), ds[varname].sel(ensemble=ens_r))
    lp.compare_plot(
        metrics.prob_negative_orig_spatial,
        metrics.prob_negative_compressed_spatial,
        'PRECT',
        method_str,
        'P(neg rainfall)',
        'P(neg rainfall)',
        'binary',
    )
