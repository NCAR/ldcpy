from unittest import TestCase

import pytest

import ldcpy

ds = ldcpy.open_datasets(
    ['SST'],
    [
        'data/pop/pop.SST.100days.nc',
        'data/pop/zfp1.0.pop.SST.100days.nc',
        'data/pop/zfp1e-1.pop.SST.100days.nc',
        'data/pop/zfp1e-2.pop.SST.100days.nc',
    ],
    ['orig', 'recon', 'recon2', 'recon3'],
)


class TestPlotPop(TestCase):
    """
    Note: The tests in this class currentlytest the plot() function in ldcpy.plot for a variety of different
    parameters on ocean (POP) data
    """

    def test_mean(self):
        ldcpy.plot(ds, 'SST', sets=['orig', 'recon'], calc='mean', vert_plot=True, tex_format=False)
        self.assertTrue(True)

    def test_standardized_mean(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon2'],
            calc='standardized_mean',
            plot_type='time_series',
            lat=90,
            lon=0,
            calc_type='diff',
            vert_plot=True,
            tex_format=False,
        )
        self.assertTrue(True)

    def test_mean_compare(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon2', 'recon3'],
            calc='mean',
            plot_type='spatial',
            tex_format=False,
        )
        self.assertTrue(True)

    def test_lag1(self):
        ldcpy.plot(
            ds, 'SST', sets=['orig', 'recon'], calc='lag1', plot_type='spatial', tex_format=False
        )
        self.assertTrue(True)

    def test_lag1_first_difference(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon2'],
            calc='lag1_first_difference',
            plot_type='spatial',
            tex_format=False,
        )
        self.assertTrue(True)

    def test_annual_harmonic(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='ann_harmonic_ratio',
            calc_type='metric_of_diff',
            tex_format=False,
        )
        self.assertTrue(True)

    def test_pooled_variance_ratio(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon', 'recon2'],
            scale='log',
            calc='pooled_var_ratio',
            calc_type='diff',
            tex_format=False,
        )

    def test_std_dev_compare(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='std',
            color='cmo.thermal',
            plot_type='spatial',
            tex_format=False,
        )
        self.assertTrue(True)

    def test_mean_diff(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='mean',
            calc_type='diff',
            transform='log',
            tex_format=False,
        )
        self.assertTrue(True)

    def test_prob_negative_log_compare(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='prob_negative',
            color='coolwarm',
            transform='log',
            plot_type='spatial',
            tex_format=False,
        )
        self.assertTrue(True is True)

    def test_odds_positive_grouped(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='odds_positive',
            plot_type='time_series',
            group_by='time.month',
            calc_ssim=True,
            vert_plot=True,
        )
        self.assertTrue(True)

    def test_prob_neg_compare(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='prob_negative',
            color='binary',
            plot_type='spatial',
            tex_format=False,
        )
        self.assertTrue(True is True)

    def test_mean_abs_diff_time_series(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon', 'recon2'],
            calc='mean_abs',
            calc_type='diff',
            plot_type='time_series',
            tex_format=False,
        )
        self.assertTrue(True is True)

    @pytest.mark.nonsequential
    def test_mean_diff_time_series_subset(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['recon', 'orig'],
            calc_type='diff',
            calc='mean',
            plot_type='time_series',
            subset='first50',
            lat=44.56,
            lon=-123.26,
            tex_format=False,
        )
        self.assertTrue(True)

    def test_subset_lat_lon_ratio_time_series(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='mean',
            calc_type='ratio',
            group_by=None,
            subset='first50',
            lat=44.76,
            lon=-93.75,
            plot_type='time_series',
            tex_format=False,
        )
        self.assertTrue(True is True)

    def test_periodogram_grouped(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='mean',
            calc_type='raw',
            plot_type='periodogram',
            group_by='time.dayofyear',
            tex_format=False,
        )
        self.assertTrue(True is True)

    def test_winter_histogram(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='mean',
            calc_type='diff',
            subset='winter',
            plot_type='histogram',
            group_by='time.dayofyear',
            tex_format=False,
        )
        self.assertTrue(True is True)

    def test_zscore_plot(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc_type='metric_of_diff',
            calc='zscore',
            tex_format=False,
        )
        self.assertTrue(True is True)

    def test_ssim(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='mean',
            plot_type='spatial',
            calc_ssim=True,
            tex_format=False,
        )

    def test_mae_max_day(self):
        ldcpy.plot(ds, 'SST', sets=['orig'], calc='mae_day_max', tex_format=False)

    def test_std_by_month(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='mean',
            plot_type='time_series',
            group_by='time.month',
            calc_type='diff',
            tex_format=False,
        )
        self.assertTrue(True)

    # Time series plot of first seven TS mean data points for ds orig dataset
    def test_mean_start_end(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig'],
            calc='mean',
            start=0,
            end=8,
            plot_type='time_series',
            tex_format=False,
        )
        self.assertTrue(True)

    def test_mean_time_series(self):
        ldcpy.plot(ds, 'SST', sets=['orig'], calc='mean', plot_type='time_series', tex_format=False)
        self.assertTrue(True)

    def test_periodogram(self):
        ldcpy.plot(ds, 'SST', sets=['orig'], calc='mean', plot_type='periodogram', tex_format=False)
        self.assertTrue(True)

    def test_plot_multiple_time_series(self):
        ldcpy.plot(
            ds,
            'SST',
            sets=['orig', 'recon'],
            calc='mean',
            plot_type='time_series',
            tex_format=False,
        )
        self.assertTrue(True)
