from unittest import TestCase

import numpy as np

import ldcpy
import ldcpy.plot as lp

ds = ldcpy.open_datasets(
    [
        'data/cam-fv/orig.TS.100days.nc',
        'data/cam-fv/zfp1.0.TS.100days.nc',
        'data/cam-fv/zfp1e-1.TS.100days.nc',
    ],
    ['orig', 'recon', 'recon2'],
)
ds2 = ldcpy.open_datasets(
    [
        'data/cam-fv/orig.PRECT.100days.nc',
        'data/cam-fv/zfp1e-7.PRECT.100days.nc',
        'data/cam-fv/zfp1e-11.PRECT.100days.nc',
    ],
    ['orig', 'recon', 'recon_2'],
)
ds3 = ldcpy.open_datasets(['data/cam-fv/cam-fv.T.6months.nc'], ['orig'])


class TestPlot(TestCase):
    """
    Note: The tests in this class currently only test the plot() function in ldcpy.plot for a variety of different
    parameters. Tests still need to be written for the methods in the plot.py class.
    """

    def test_mean(self):
        lp.plot(ds, 'TS', 'orig', ens_r='recon', metric='mean')
        self.assertTrue(True is True)

    def test_prob_neg(self):
        lp.plot(ds2, 'PRECT', 'orig', ens_r='recon', metric='prob_negative')
        self.assertTrue(True is True)

    def test_std_dev_compare(self):
        lp.plot(
            ds,
            'TS',
            'orig',
            ens_r='recon',
            metric='std',
            color='cmo.thermal',
            plot_type='spatial_comparison',
        )
        self.assertTrue(True is True)

    def test_mean_diff(self):
        lp.plot(ds, 'TS', 'orig', ens_r='recon', metric='mean', metric_type='diff')
        self.assertTrue(True is True)

    def test_prob_negative_log_compare(self):
        lp.plot(
            ds,
            'TS',
            'orig',
            ens_r='recon',
            metric='prob_negative',
            color='coolwarm',
            transform='log',
            plot_type='spatial_comparison',
        )
        self.assertTrue(True is True)

    def test_log_odds_positive_compare(self):
        lp.plot(
            ds2,
            'PRECT',
            'orig',
            ens_r='recon',
            metric='odds_positive',
            metric_type='ratio',
            transform='log',
            color='cmo.thermal',
        )
        self.assertTrue(True is True)

    def test_prob_neg_compare(self):
        lp.plot(
            ds2,
            'PRECT',
            'orig',
            ens_r='recon',
            metric='prob_negative',
            color='binary',
            plot_type='spatial_comparison',
        )
        self.assertTrue(True is True)

    def test_mean_abs_diff_time_series(self):
        lp.plot(
            ds,
            'TS',
            'orig',
            ens_r='recon',
            group_by='time.dayofyear',
            metric='mean_abs',
            metric_type='diff',
            plot_type='time_series',
        )
        self.assertTrue(True is True)

    def test_subset_lat_lon_ratio_time_series(self):
        lp.plot(
            ds2,
            'PRECT',
            'orig',
            metric='mean',
            ens_r='recon',
            metric_type='ratio',
            group_by=None,
            subset='first50',
            lat=44.76,
            lon=-93.75,
            plot_type='time_series',
        )
        self.assertTrue(True is True)

    def test_periodogram_grouped(self):
        lp.plot(
            ds2,
            'PRECT',
            'orig',
            ens_r='recon',
            metric='mean',
            metric_type='raw',
            plot_type='periodogram',
            standardized_err=False,
            group_by='time.dayofyear',
        )
        self.assertTrue(True is True)

    def test_winter_histogram(self):
        lp.plot(
            ds2,
            'PRECT',
            'orig',
            ens_r='recon',
            metric='mean',
            metric_type='diff',
            subset='winter',
            plot_type='histogram',
            group_by='time.dayofyear',
        )
        self.assertTrue(True is True)

    def test_time_series_single_point_3d_data(self):
        lp.plot(ds3, 'T', 'orig', metric='mean', plot_type='time_series', group_by='time.day')
        self.assertTrue(True is True)

    def test_zscore_plot(self):
        lp.plot(ds, 'TS', 'orig', ens_r='recon', metric_type='metric_of_diff', metric='zscore')
        self.assertTrue(True is True)
