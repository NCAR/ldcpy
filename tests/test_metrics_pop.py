from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ldcpy
from ldcpy.metrics import DatasetMetrics, DiffMetrics

lats = np.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 3.9], [1, 1.9, 2.9, 3.9]])
lons = np.array([[3.0, 4.0, 5.0, 6.0], [3.1, 4.1, 5.1, 5.9], [3, 3.9, 4.9, 5.9]])
times = pd.date_range('2000-01-01', periods=5)

pop_data = np.arange(1, 61).reshape(3, 4, 5)
test_data = xr.DataArray(
    pop_data,
    coords={
        'time': times,
        'TLAT': (('nlat', 'nlon'), lats, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        'TLON': (('nlat', 'nlon'), lons, {'standard_name': 'longitude', 'units': 'degrees_east'}),
    },
    dims=['nlat', 'nlon', 'time'],
    attrs={'long_name': 'Surface Potential'},
)

pop_data2 = np.arange(0, 60).reshape(3, 4, 5)
test_data_2 = xr.DataArray(
    pop_data2,
    coords={
        'time': times,
        'TLAT': (('nlat', 'nlon'), lats, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        'TLON': (('nlat', 'nlon'), lons, {'standard_name': 'longitude', 'units': 'degrees_east'}),
    },
    dims=['nlat', 'nlon', 'time'],
    attrs={'long_name': 'Surface Potential'},
)

test_overall_metrics = ldcpy.DatasetMetrics(test_data, ['time', 'nlat', 'nlon'])
test_spatial_metrics = ldcpy.DatasetMetrics(test_data, ['time'])
test_time_series_metrics = ldcpy.DatasetMetrics(test_data, ['nlat', 'nlon'])
test_diff_metrics = ldcpy.DiffMetrics(test_data, test_data_2, ['time', 'nlat', 'nlon'])


class TestErrorMetricsPOP(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mylat = np.array([[1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 3.9], [1, 1.9, 2.9, 3.9]])
        mylon = np.array([[3.0, 4.0, 5.0, 6.0], [3.1, 4.1, 5.1, 5.9], [3, 3.9, 4.9, 5.9]])
        mydata = np.arange(0, 12, dtype='int64').reshape(3, 4)
        myzero = np.zeros(12, dtype='int64').reshape(3, 4)
        cls._samples = [
            {
                'measured': (
                    xr.DataArray(
                        mydata,
                        coords={
                            'TLAT': (
                                ('nlat', 'nlon'),
                                mylat,
                                {'standard_name': 'latitude', 'units': 'degrees_north'},
                            ),
                            'TLON': (
                                ('nlat', 'nlon'),
                                mylon,
                                {'standard_name': 'longitude', 'units': 'degrees_east'},
                            ),
                        },
                        dims=['nlat', 'nlon'],
                    )
                ),
                'observed': (
                    xr.DataArray(
                        mydata,
                        coords={
                            'TLAT': (
                                ('nlat', 'nlon'),
                                mylat,
                                {'standard_name': 'latitude', 'units': 'degrees_north'},
                            ),
                            'TLON': (
                                ('nlat', 'nlon'),
                                mylon,
                                {'standard_name': 'longitude', 'units': 'degrees_east'},
                            ),
                        },
                        dims=['nlat', 'nlon'],
                    )
                ),
                'expected_error': (
                    xr.DataArray(
                        myzero,
                        coords={
                            'TLAT': (
                                ('nlat', 'nlon'),
                                mylat,
                                {'standard_name': 'latitude', 'units': 'degrees_north'},
                            ),
                            'TLON': (
                                ('nlat', 'nlon'),
                                mylon,
                                {'standard_name': 'longitude', 'units': 'degrees_east'},
                            ),
                        },
                        dims=['nlat', 'nlon'],
                    )
                ),
            }
        ]

    def test_creation_01(self):
        DiffMetrics(
            xr.DataArray(self._samples[0]['observed']),
            xr.DataArray(self._samples[0]['measured']),
            [],
        )

    def test_error_01(self):
        em = DatasetMetrics(
            xr.DataArray(self._samples[0]['observed']) - xr.DataArray(self._samples[0]['measured']),
            [],
        )

        self.assertTrue((self._samples[0]['expected_error'] == em.sum).all())

    def test_mean_error_01(self):
        em = DatasetMetrics(
            xr.DataArray(self._samples[0]['observed']) - xr.DataArray(self._samples[0]['measured']),
            [],
        )
        self.assertTrue(em.mean.all() == 0.0)

    def test_mean_error_02(self):
        em = DatasetMetrics(
            xr.DataArray(self._samples[0]['observed'] - xr.DataArray(self._samples[0]['measured'])),
            [],
        )

        self.assertTrue(em.mean.all() == 0.0)

    def test_dim_names(self):
        self.assertTrue(test_spatial_metrics._lat_dim_name == 'nlat')
        self.assertTrue(test_spatial_metrics._lon_dim_name == 'nlon')
        self.assertTrue(test_spatial_metrics._time_dim_name == 'time')

    def test_TS_02(self):
        import xarray as xr
        import zfpy

        ds = xr.open_dataset('data/pop/pop.SST.100days.nc')

        SST = ds.SST

        print(type(SST))

    def test_mean(self):
        self.assertTrue(test_overall_metrics.mean == 30.5)

    def test_mean_abs(self):
        self.assertTrue(test_overall_metrics.mean_abs == 30.5)

    def test_mean_squared(self):
        self.assertTrue(np.isclose(test_overall_metrics.mean_squared, 930.25, rtol=1e-09))

    def test_min_abs(self):
        self.assertTrue(test_overall_metrics.min_abs == 1)

    def test_max_abs(self):
        self.assertTrue(test_overall_metrics.max_abs == 60)

    def test_min_val(self):
        self.assertTrue(test_overall_metrics.min_val == 1)

    def test_max_val(self):
        self.assertTrue(test_overall_metrics.max_val == 60)

    def test_ns_con_var(self):
        self.assertTrue(test_overall_metrics.ns_con_var == 400)

    def test_ew_con_var(self):
        self.assertTrue(test_overall_metrics.ew_con_var == 75)

    #    def test_odds_positive(self):
    #        self.assertTrue(np.isclose(test_overall_metrics.odds_positive, 0.98019802, rtol=1e-09))

    def test_prob_negative(self):
        self.assertTrue(test_overall_metrics.prob_negative == 0)

    def test_prob_positive(self):
        self.assertTrue(test_overall_metrics.prob_positive == 1.0)

    def test_dyn_range(self):
        self.assertTrue(test_overall_metrics.dyn_range == 59)

    def test_median(self):
        self.assertTrue(test_overall_metrics.get_metric('quantile', 0.5) == 30.5)

    def test_rms(self):
        self.assertTrue(np.isclose(test_overall_metrics.get_metric('rms'), 35.07373186, rtol=1e-09))

    def test_std(self):
        self.assertTrue(np.isclose(test_overall_metrics.get_metric('std'), 17.4642492, rtol=1e-09))

    def test_sum(self):
        self.assertTrue(test_overall_metrics.get_metric('sum') == 1830)

    def test_variance(self):
        self.assertTrue(
            np.isclose(test_overall_metrics.get_metric('variance'), 299.91666667, rtol=1e-6)
        )

    def test_zscore(self):
        self.assertTrue(
            np.isclose(test_overall_metrics.get_metric('zscore'), 3.90512484, rtol=1e-09)
        )

    def test_mean_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('mean')
                == np.array(
                    [
                        [3.0, 8.0, 13.0, 18.0],
                        [23.0, 28.0, 33.0, 38.0],
                        [43.0, 48.0, 53.0, 58.0],
                    ]
                )
            ).all()
        )

    def test_mean_abs_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('mean_abs')
                == np.array(
                    [
                        [3.0, 8.0, 13.0, 18.0],
                        [23.0, 28.0, 33.0, 38.0],
                        [43.0, 48.0, 53.0, 58.0],
                    ]
                )
            ).all()
        )

    def test_mean_squared_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_metrics.get_metric('mean_squared'),
                np.array(
                    [
                        [9.0, 64.0, 169.0, 324.0],
                        [529.0, 784.0, 1089.0, 1444.0],
                        [1849.0, 2304.0, 2809.0, 3364.0],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_min_abs_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('min_abs')
                == np.array(
                    [[1.0, 6.0, 11.0, 16.0], [21.0, 26.0, 31.0, 36.0], [41.0, 46.0, 51.0, 56.0]]
                )
            ).all()
        )

    def test_max_abs_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('max_abs')
                == np.array(
                    [[5.0, 10.0, 15.0, 20.0], [25.0, 30.0, 35.0, 40.0], [45.0, 50.0, 55.0, 60.0]]
                )
            ).all()
        )

    def test_min_val_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('min_val')
                == np.array(
                    [[1.0, 6.0, 11.0, 16.0], [21.0, 26.0, 31.0, 36.0], [41.0, 46.0, 51.0, 56.0]]
                )
            ).all()
        )

    def test_max_val_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('max_val')
                == np.array(
                    [[5.0, 10.0, 15.0, 20.0], [25.0, 30.0, 35.0, 40.0], [45.0, 50.0, 55.0, 60.0]]
                )
            ).all()
        )

        def test_ns_con_var_spatial(self):
            self.assertTrue(
                (
                    test_spatial_metrics.get_metric('ns_con_var')
                    == np.array([[400.0, 400.0, 400.0, 400.0], [400.0, 400.0, 400.0, 400.0]])
                ).all()
            )

    def test_odds_positive_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_metrics.get_metric('odds_positive'),
                np.array(
                    [
                        [np.inf, np.inf, np.inf, np.inf],
                        [np.inf, np.inf, np.inf, np.inf],
                        [np.inf, np.inf, np.inf, np.inf],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_prob_positive_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_metrics.get_metric('prob_positive'),
                np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]),
                rtol=1e-09,
            ).all()
        )

    def test_prob_negative_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_metrics.get_metric('prob_negative'),
                np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
                rtol=1e-09,
            ).all()
        )

    def test_median_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('quantile', 0.5)
                == np.array(
                    [[3.0, 8.0, 13.0, 18.0], [23.0, 28.0, 33.0, 38.0], [43.0, 48.0, 53.0, 58.0]]
                )
            ).all()
        )

    def test_rms_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_metrics.get_metric('rms'),
                np.array(
                    [
                        [3.31662479, 8.1240384, 13.07669683, 18.05547009],
                        [23.04343724, 28.03569154, 33.03028913, 38.02630668],
                        [43.02324953, 48.02082881, 53.01886457, 58.01723882],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_std_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_metrics.get_metric('std'),
                np.array(
                    [
                        [1.58113883, 1.58113883, 1.58113883, 1.58113883],
                        [1.58113883, 1.58113883, 1.58113883, 1.58113883],
                        [1.58113883, 1.58113883, 1.58113883, 1.58113883],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_sum_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('sum')
                == np.array(
                    [
                        [15.0, 40.0, 65.0, 90.0],
                        [115.0, 140.0, 165.0, 190.0],
                        [215.0, 240.0, 265.0, 290.0],
                    ]
                )
            ).all()
        )

    def test_variance_spatial(self):
        self.assertTrue(
            (
                test_spatial_metrics.get_metric('variance')
                == np.array([[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]])
            ).all()
        )

    def test_zscore_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_metrics.get_metric('zscore'),
                np.array(
                    [
                        [4.24264069, 11.3137085, 18.38477631, 25.45584412],
                        [32.52691193, 39.59797975, 46.66904756, 53.74011537],
                        [60.81118318, 67.88225099, 74.95331881, 82.02438662],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

        def test_ew_con_var_spatial(self):
            self.assertTrue(
                (
                    test_spatial_metrics.get_metric('ew_con_var')
                    == np.array(
                        [
                            [25.0, 25.0, 25.0, 225.0],
                            [25.0, 25.0, 25.0, 225.0],
                            [25.0, 25.0, 25.0, 225.0],
                        ]
                    )
                ).all()
            )

    def test_mean_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('mean'),
                np.array([28.5, 29.5, 30.5, 31.5, 32.5]),
                rtol=1e-09,
            ).all()
        )

    def test_mean_abs_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('mean_abs'),
                np.array([28.5, 29.5, 30.5, 31.5, 32.5]),
                rtol=1e-09,
            ).all()
        )

    def test_mean_squared_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('mean_squared'),
                np.array([812.25, 870.25, 930.25, 992.25, 1056.25]),
                rtol=1e-09,
            ).all()
        )

    def test_max_abs_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('max_abs'),
                np.array([56.0, 57.0, 58.0, 59.0, 60.0]),
                rtol=1e-09,
            ).all()
        )

    def test_max_val_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('max_val'),
                np.array([56.0, 57.0, 58.0, 59.0, 60.0]),
                rtol=1e-09,
            ).all()
        )

    def test_min_abs_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('min_abs'),
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                rtol=1e-09,
            ).all()
        )

    def test_min_val_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('min_val'),
                np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
                rtol=1e-09,
            ).all()
        )

        def test_ns_con_var_time_series(self):
            self.assertTrue(
                np.isclose(
                    test_time_series_metrics.get_metric('ns_con_var'),
                    np.array([400.0, 400.0, 400.0, 400.0, 400.0]),
                    rtol=1e-09,
                ).all()
            )

    def test_odds_positive_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('odds_positive'),
                np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
                rtol=1e-09,
            ).all()
        )

    def test_prob_negative_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('prob_negative'),
                np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                rtol=1e-09,
            ).all()
        )

    def test_prob_positive_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('prob_positive'),
                np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
                rtol=1e-09,
            ).all()
        )

    def test_median_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('quantile', 0.5),
                np.array([28.5, 29.5, 30.5, 31.5, 32.5]),
                rtol=1e-09,
            ).all()
        )

    def test_rms_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('rms'),
                np.array([33.31916365, 34.17845325, 35.0452089, 35.91889011, 36.79900361]),
                rtol=1e-09,
            ).all()
        )

    def test_std_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('std'),
                np.array([18.02775638, 18.02775638, 18.02775638, 18.02775638, 18.02775638]),
                rtol=1e-09,
            ).all()
        )

    def test_sum_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('sum'),
                np.array([342.0, 354.0, 366.0, 378.0, 390.0]),
                rtol=1e-09,
            ).all()
        )

    def test_variance_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('variance'),
                np.array([297.91666667, 297.91666667, 297.91666667, 297.91666667, 297.91666667]),
                rtol=1e-09,
            ).all()
        )

    def test_zscore_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_metrics.get_metric('zscore'),
                np.array([3.53498994, 3.65902467, 3.7830594, 3.90709414, 4.03112887]),
                rtol=1e-09,
            ).all()
        )

        def test_ew_con_var_time_series(self):
            self.assertTrue(
                np.isclose(
                    test_time_series_metrics.get_metric('ew_con_var'),
                    np.array([75.0, 75.0, 75.0, 75.0, 75.0]),
                    rtol=1e-09,
                ).all()
            )

    def test_diff_pcc(self):
        self.assertTrue(
            np.isclose(
                test_diff_metrics.get_diff_metric('pearson_correlation_coefficient'),
                np.array(1),
                rtol=1e-09,
            ).all()
        )

    def test_diff_ksp(self):
        self.assertTrue(
            np.isclose(
                test_diff_metrics.get_diff_metric('ks_p_value'),
                np.array(1.0),
                rtol=1e-09,
            ).all()
        )

    def test_diff_covariance(self):
        self.assertTrue(
            np.isclose(
                test_diff_metrics.get_diff_metric('covariance'),
                np.array(299.91666667),
                rtol=1e-09,
            ).all()
        )

    def test_diff_normalized_max_pointwise_error(self):
        self.assertTrue(
            np.isclose(
                test_diff_metrics.get_diff_metric('n_emax'),
                np.array(0.01694915),
                rtol=1e-09,
            ).all()
        )

    def test_diff_normalized_root_mean_squared(self):
        self.assertTrue(
            np.isclose(
                test_diff_metrics.get_diff_metric('n_rms'),
                np.array(0.01694915),
                rtol=1e-09,
            ).all()
        )
