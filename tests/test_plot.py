from unittest import TestCase

import pytest

import ldcpy

ds = ldcpy.open_datasets(
    ['TS'],
    [
        'data/cam-fv/orig.TS.100days.nc',
        'data/cam-fv/zfp1.0.TS.100days.nc',
        'data/cam-fv/zfp1e-1.TS.100days.nc',
    ],
    ['orig', 'recon', 'recon2'],
)
ds2 = ldcpy.open_datasets(
    ['PRECT'],
    [
        'data/cam-fv/orig.PRECT.60days.nc',
        'data/cam-fv/zfp1e-7.PRECT.60days.nc',
        'data/cam-fv/zfp1e-11.PRECT.60days.nc',
    ],
    ['orig', 'recon', 'recon_2'],
)
ds3 = ldcpy.open_datasets(['T'], ['data/cam-fv/cam-fv.T.3months.nc'], ['orig'])


class TestPlot(TestCase):
    """
    Note: The tests in this class currently only test the plot() function in ldcpy.plot for a variety of different
    parameters. Tests still need to be written for the methods in the plot.py class.
    """

    @pytest.mark.nonsequential
    def test_mean(self):
        ldcpy.plot(ds, 'TS', set1='orig', set2='recon', metric='mean')
        self.assertTrue(True)

    @pytest.mark.nonsequential
    def test_prob_neg(self):
        ldcpy.plot(ds2, 'PRECT', set1='orig', set2='recon', metric='prob_negative')
        self.assertTrue(True)

    @pytest.mark.nonsequential
    def test_mean_compare(self):
        ldcpy.plot(
            ds, 'TS', set1='orig', metric='mean', set2='recon', plot_type='spatial_comparison',
        )
        self.assertTrue(True)

    @pytest.mark.nonsequential
    def test_std_dev_compare(self):
        ldcpy.plot(
            ds,
            'TS',
            set1='orig',
            set2='recon',
            metric='std',
            color='cmo.thermal',
            plot_type='spatial_comparison',
        )
        self.assertTrue(True)

    @pytest.mark.nonsequential
    def test_mean_diff(self):
        ldcpy.plot(ds, 'TS', set1='orig', set2='recon', metric='mean', metric_type='diff')
        self.assertTrue(True)

    @pytest.mark.nonsequential
    def test_mean_diff_standardized(self):
        with pytest.raises(ValueError):
            ldcpy.plot(
                ds,
                'TS',
                set1='orig',
                set2='recon',
                metric='mean',
                metric_type='diff',
                standardized_err=True,
            )
        self.assertTrue(True)

    @pytest.mark.nonsequential
    def test_prob_negative_log_compare(self):
        ldcpy.plot(
            ds,
            'TS',
            set1='orig',
            set2='recon',
            metric='prob_negative',
            color='coolwarm',
            transform='log',
            plot_type='spatial_comparison',
        )
        self.assertTrue(True is True)

    @pytest.mark.nonsequential
    def test_log_odds_positive_compare(self):
        ldcpy.plot(
            ds2,
            'PRECT',
            set1='orig',
            set2='recon',
            metric='odds_positive',
            metric_type='ratio',
            transform='log',
            color='cmo.thermal',
        )
        self.assertTrue(True is True)

    @pytest.mark.nonsequential
    def test_prob_neg_compare(self):
        ldcpy.plot(
            ds2,
            'PRECT',
            set1='orig',
            set2='recon',
            metric='prob_negative',
            color='binary',
            plot_type='spatial_comparison',
        )
        self.assertTrue(True is True)

    @pytest.mark.sequential
    def test_mean_abs_diff_time_series(self):
        ldcpy.plot(
            ds,
            'TS',
            set1='orig',
            set2='recon',
            group_by='time.dayofyear',
            metric='mean_abs',
            metric_type='diff',
            plot_type='time_series',
        )
        self.assertTrue(True is True)

    @pytest.mark.sequential
    def test_mean_diff_time_series_subset(self):
        ldcpy.plot(
            ds2,
            'PRECT',
            set1='recon',
            set2='orig',
            metric_type='diff',
            metric='mean',
            plot_type='time_series',
            subset='first50',
            lat=44.56,
            lon=-123.26,
        )
        self.assertTrue(True)

    @pytest.mark.sequential
    def test_subset_lat_lon_ratio_time_series(self):
        ldcpy.plot(
            ds2,
            'PRECT',
            set1='orig',
            metric='mean',
            set2='recon',
            metric_type='ratio',
            group_by=None,
            subset='first50',
            lat=44.76,
            lon=-93.75,
            plot_type='time_series',
        )
        self.assertTrue(True is True)

    @pytest.mark.nonsequential
    def test_periodogram_grouped(self):
        ldcpy.plot(
            ds2,
            'PRECT',
            set1='orig',
            set2='recon',
            metric='mean',
            metric_type='raw',
            plot_type='periodogram',
            standardized_err=False,
            group_by='time.dayofyear',
        )
        self.assertTrue(True is True)

    @pytest.mark.nonsequential
    def test_winter_histogram(self):
        ldcpy.plot(
            ds2,
            'PRECT',
            set1='orig',
            set2='recon',
            metric='mean',
            metric_type='diff',
            subset='winter',
            plot_type='histogram',
            group_by='time.dayofyear',
        )
        self.assertTrue(True is True)

    @pytest.mark.sequential
    def test_time_series_single_point_3d_data(self):
        ldcpy.plot(
            ds3, 'T', set1='orig', metric='mean', plot_type='time_series', group_by='time.day',
        )
        self.assertTrue(True is True)

    @pytest.mark.nonsequential
    def test_zscore_plot(self):
        ldcpy.plot(
            ds, 'TS', set1='orig', set2='recon', metric_type='metric_of_diff', metric='zscore',
        )
        self.assertTrue(True is True)

    @pytest.mark.nonsequential
    def test_mean_3d(self):
        ldcpy.plot(ds3, 'T', set1='orig', metric='mean', lev=29)
        self.assertTrue(True)

    @pytest.mark.sequential
    def test_std_by_month(self):
        ldcpy.plot(
            ds, 'TS', set1='orig', metric='std', plot_type='time_series', group_by='time.month'
        )
        self.assertTrue(True)

    # Time series plot of first seven TS mean data points for ds orig dataset
    @pytest.mark.sequential
    def test_mean_start_end(self):
        ldcpy.plot(ds, 'TS', set1='orig', metric='mean', start=0, end=8, plot_type='time_series')
        self.assertTrue(True)
