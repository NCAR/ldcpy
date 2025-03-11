from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ldcpy
from ldcpy.calcs import Datasetcalcs, Diffcalcs

ds = ldcpy.open_datasets(
    'cam-fv',
    ['TS'],
    [
        'data/cam-fv/orig.TS.100days.nc',
        'data/cam-fv/zfp1.0.TS.100days.nc',
        'data/cam-fv/zfp1e-1.TS.100days.nc',
    ],
    ['orig', 'recon', 'recon2'],
)

times = pd.date_range('2000-01-01', periods=10)
lats = [0, 1, 2, 3]
lons = [0, 1, 2, 3, 4]
test_data = xr.DataArray(
    np.arange(-100.0, 100.0).reshape(4, 5, 10),
    coords=[
        ('lat', lats, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        ('lon', lons, {'standard_name': 'longitude', 'units': 'degrees_east'}),
        ('time', times, {'standard_name': 'time'}),
    ],
    dims=['lat', 'lon', 'time'],
)
test_data_2 = xr.DataArray(
    np.arange(-99.0, 101.0).reshape(4, 5, 10),
    coords=[
        ('lat', lats, {'standard_name': 'latitude', 'units': 'degrees_north'}),
        ('lon', lons, {'standard_name': 'longitude', 'units': 'degrees_east'}),
        ('time', times, {'standard_name': 'time'}),
    ],
    dims=['lat', 'lon', 'time'],
)


d = ds['TS'].sel(collection='orig')

ds_pointwise_calcs = ldcpy.Datasetcalcs(d, 'cam-fv', [], weighted=False)
ds_spatial_calcs = ldcpy.Datasetcalcs(d, 'cam-fv', ['time'], weighted=False)

calc_ds = ds_pointwise_calcs.get_calc_ds('w_e_derivative', 'd')

test_pointwise_calcs = ldcpy.Datasetcalcs(test_data, 'cam-fv', [], weighted=False)
test_overall_calcs = ldcpy.Datasetcalcs(test_data, 'cam-fv', ['time', 'lat', 'lon'], weighted=False)
test_spatial_calcs = ldcpy.Datasetcalcs(test_data, 'cam-fv', ['time'], weighted=False)
test_time_series_calcs = ldcpy.Datasetcalcs(test_data, 'cam-fv', ['lat', 'lon'], weighted=False)
test_diff_calcs = ldcpy.Diffcalcs(
    test_data, test_data_2, 'cam-fv', ['time', 'lat', 'lon'], weighted=False
)


class TestErrorcalcs(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        mylon = np.arange(0, 10)
        mylat = np.arange(0, 10)
        mydata = np.arange(0, 100, dtype='int64').reshape(10, 10)
        myzero = np.zeros(100, dtype='int64').reshape(10, 10)
        cls._samples = [
            {
                'measured': (
                    xr.DataArray(
                        mydata,
                        coords=[
                            ('lat', mylat, {'standard_name': 'latitude', 'units': 'degrees_north'}),
                            ('lon', mylon, {'standard_name': 'longitude', 'units': 'degrees_east'}),
                        ],
                        dims=['lat', 'lon'],
                    )
                ),
                'observed': (
                    xr.DataArray(
                        mydata,
                        coords=[
                            ('lat', mylat, {'standard_name': 'latitude', 'units': 'degrees_north'}),
                            ('lon', mylon, {'standard_name': 'longitude', 'units': 'degrees_east'}),
                        ],
                        dims=['lat', 'lon'],
                    )
                ),
                'expected_error': (
                    xr.DataArray(
                        myzero,
                        coords=[
                            ('lat', mylat, {'standard_name': 'latitude', 'units': 'degrees_north'}),
                            ('lon', mylon, {'standard_name': 'longitude', 'units': 'degrees_east'}),
                        ],
                        dims=['lat', 'lon'],
                    )
                ),
            }
        ]

    def test_creation_01(self):
        Diffcalcs(
            xr.DataArray(self._samples[0]['observed']),
            xr.DataArray(self._samples[0]['measured']),
            'cam-fv',
        )

    def test_error_01(self):
        em = Datasetcalcs(
            xr.DataArray(self._samples[0]['observed']) - xr.DataArray(self._samples[0]['measured']),
            'cam-fv',
            [],
            weighted=False,
        )

        self.assertTrue((self._samples[0]['expected_error'] == em.sum).all())

    def test_mean_error_01(self):
        em = Datasetcalcs(
            xr.DataArray(self._samples[0]['observed']) - xr.DataArray(self._samples[0]['measured']),
            'cam-fv',
            [],
            weighted=False,
        )
        self.assertTrue(em.mean.all() == 0.0)

    def test_mean_error_02(self):
        em = Datasetcalcs(
            xr.DataArray(self._samples[0]['observed'] - xr.DataArray(self._samples[0]['measured'])),
            'cam-fv',
            [],
            weighted=False,
        )

        self.assertTrue(em.mean.all() == 0.0)

        em.mean_error = 42.0

        self.assertTrue(em.mean.all() == 0.0)

    def test_dim_names(self):
        self.assertTrue(test_spatial_calcs._lat_dim_name == 'lat')
        self.assertTrue(test_spatial_calcs._lon_dim_name == 'lon')
        self.assertTrue(test_spatial_calcs._time_dim_name == 'time')

    def test_TS_02(self):
        import xarray as xr

        # import zfpy
        ds = xr.open_dataset('data/cam-fv/orig.TS.100days.nc')

        TS = ds.TS
        print(type(TS))

    def test_calc_ds(self):
        ds_pointwise_calcs.get_calc_ds('w_e_derivative', 'test_deriv')
        self.assertTrue(True)

    def test_mean(self):
        self.assertTrue(test_overall_calcs.mean == -0.5)

    def test_magnitude_range(self):
        value = float(test_overall_calcs.magnitude_range.values)
        # print('VALUE = ', value)
        self.assertTrue(np.isclose(value, 2, rtol=1e-04))

    def test_mean_abs(self):
        self.assertTrue(test_overall_calcs.mean_abs == 50)

    def test_mean_squared(self):
        self.assertTrue(np.isclose(test_overall_calcs.mean_squared, 0.25, rtol=1e-09))

    def test_most_repeated(self):
        self.assertTrue(test_overall_calcs.most_repeated == -100)

    def test_min_abs(self):
        self.assertTrue(test_overall_calcs.min_abs == 0)

    def test_max_abs(self):
        self.assertTrue(test_overall_calcs.max_abs == 100)

    def test_min_val(self):
        self.assertTrue(test_overall_calcs.min_val == -100)

    def test_max_val(self):
        self.assertTrue(test_overall_calcs.max_val == 99)

    def test_ns_con_var(self):
        self.assertTrue(test_overall_calcs.ns_con_var == 2500)  # is this right?

    def test_ew_con_var(self):
        self.assertTrue(test_overall_calcs.ew_con_var == 400)  # is this right?

    def test_odds_positive(self):
        self.assertTrue(np.isclose(test_overall_calcs.odds_positive, 0.98019802, rtol=1e-09))

    def test_prob_negative(self):
        self.assertTrue(test_overall_calcs.prob_negative == 0.5)

    def test_prob_positive(self):
        self.assertTrue(test_overall_calcs.prob_positive == 0.495)

    def test_dyn_range(self):
        self.assertTrue(test_overall_calcs.dyn_range == 199)

    def test_median(self):
        self.assertTrue(test_overall_calcs.get_calc('quantile', 0.5) == -0.5)

    def test_rms(self):
        self.assertTrue(np.isclose(test_overall_calcs.get_calc('rms'), 57.73647028, rtol=1e-09))

    def test_std(self):
        self.assertTrue(np.isclose(test_overall_calcs.get_calc('std'), 57.87918451, rtol=1e-09))

    def test_sum(self):
        self.assertTrue(test_overall_calcs.get_calc('sum') == -100)

    def test_variance(self):
        self.assertTrue(test_overall_calcs.get_calc('variance') == 3333.25)

    def test_zscore(self):
        self.assertTrue(np.isclose(test_overall_calcs.get_calc('zscore'), -0.02731792, rtol=1e-09))

    def test_mean_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('mean')
                == np.array(
                    [
                        [-95.5, -85.5, -75.5, -65.5, -55.5],
                        [-45.5, -35.5, -25.5, -15.5, -5.5],
                        [4.5, 14.5, 24.5, 34.5, 44.5],
                        [54.5, 64.5, 74.5, 84.5, 94.5],
                    ]
                )
            ).all()
        )

    def test_mean_abs_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('mean_abs')
                == np.array(
                    [
                        [95.5, 85.5, 75.5, 65.5, 55.5],
                        [45.5, 35.5, 25.5, 15.5, 5.5],
                        [4.5, 14.5, 24.5, 34.5, 44.5],
                        [54.5, 64.5, 74.5, 84.5, 94.5],
                    ]
                )
            ).all()
        )

    def test_mean_squared_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_calcs.get_calc('mean_squared'),
                np.array(
                    [
                        [9120.25, 7310.25, 5700.25, 4290.25, 3080.25],
                        [2070.25, 1260.25, 650.25, 240.25, 30.25],
                        [20.25, 210.25, 600.25, 1190.25, 1980.25],
                        [2970.25, 4160.25, 5550.25, 7140.25, 8930.25],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_min_abs_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('min_abs')
                == np.array(
                    [
                        [91.0, 81.0, 71.0, 61.0, 51.0],
                        [41.0, 31.0, 21.0, 11.0, 1.0],
                        [0.0, 10.0, 20.0, 30.0, 40.0],
                        [50.0, 60.0, 70.0, 80.0, 90.0],
                    ]
                )
            ).all()
        )

    def test_max_abs_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('max_abs')
                == np.array(
                    [
                        [100.0, 90.0, 80.0, 70.0, 60.0],
                        [50.0, 40.0, 30.0, 20.0, 10.0],
                        [9.0, 19.0, 29.0, 39.0, 49.0],
                        [59.0, 69.0, 79.0, 89.0, 99.0],
                    ]
                )
            ).all()
        )

    def test_min_val_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('min_val')
                == np.array(
                    [
                        [-100.0, -90.0, -80.0, -70.0, -60.0],
                        [-50.0, -40.0, -30.0, -20.0, -10.0],
                        [0.0, 10.0, 20.0, 30.0, 40.0],
                        [50.0, 60.0, 70.0, 80.0, 90.0],
                    ]
                )
            ).all()
        )

    def test_max_val_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('max_val')
                == np.array(
                    [
                        [-91.0, -81.0, -71.0, -61.0, -51.0],
                        [-41.0, -31.0, -21.0, -11.0, -1.0],
                        [9.0, 19.0, 29.0, 39.0, 49.0],
                        [59.0, 69.0, 79.0, 89.0, 99.0],
                    ]
                )
            ).all()
        )

        def test_ns_con_var_spatial(self):
            self.assertTrue(
                (
                    test_spatial_calcs.get_calc('ns_con_var')
                    == np.array(
                        [
                            [2500.0, 2500.0, 2500.0, 2500.0, 2500.0],
                            [2500.0, 2500.0, 2500.0, 2500.0, 2500.0],
                            [2500.0, 2500.0, 2500.0, 2500.0, 2500.0],
                        ]
                    )
                ).all()
            )

    def test_odds_positive_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_calcs.get_calc('odds_positive'),
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [9.0, np.inf, np.inf, np.inf, np.inf],
                        [np.inf, np.inf, np.inf, np.inf, np.inf],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_prob_positive_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_calcs.get_calc('prob_positive'),
                np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.9, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_prob_negative_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_calcs.get_calc('prob_negative'),
                np.array(
                    [
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_median_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('quantile', 0.5)
                == np.array(
                    [
                        [-95.5, -85.5, -75.5, -65.5, -55.5],
                        [-45.5, -35.5, -25.5, -15.5, -5.5],
                        [4.5, 14.5, 24.5, 34.5, 44.5],
                        [54.5, 64.5, 74.5, 84.5, 94.5],
                    ]
                )
            ).all()
        )

    def test_rms_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_calcs.get_calc('rms'),
                np.array(
                    [
                        [95.54318395, 85.54823201, 75.55461601, 65.56294685, 55.57427462],
                        [45.5905692, 35.61600764, 25.66125484, 15.76388277, 6.20483682],
                        [5.33853913, 14.7817455, 24.66779277, 34.61935875, 44.59260028],
                        [54.57563559, 64.56392181, 74.55534857, 84.54880248, 94.54364072],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_std_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_calcs.get_calc('std'),
                np.array(
                    [
                        [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
                        [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
                        [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
                        [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_sum_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('sum')
                == np.array(
                    [
                        [-955.0, -855.0, -755.0, -655.0, -555.0],
                        [-455.0, -355.0, -255.0, -155.0, -55.0],
                        [45.0, 145.0, 245.0, 345.0, 445.0],
                        [545.0, 645.0, 745.0, 845.0, 945.0],
                    ]
                )
            ).all()
        )

    def test_variance_spatial(self):
        self.assertTrue(
            (
                test_spatial_calcs.get_calc('variance')
                == np.array(
                    [
                        [8.25, 8.25, 8.25, 8.25, 8.25],
                        [8.25, 8.25, 8.25, 8.25, 8.25],
                        [8.25, 8.25, 8.25, 8.25, 8.25],
                        [8.25, 8.25, 8.25, 8.25, 8.25],
                    ]
                )
            ).all()
        )

    def test_zscore_spatial(self):
        self.assertTrue(
            np.isclose(
                test_spatial_calcs.get_calc('zscore'),
                np.array(
                    [
                        [-99.74649686, -89.30183751, -78.85717815, -68.41251879, -57.96785943],
                        [-47.52320008, -37.07854072, -26.63388136, -16.189222, -5.74456265],
                        [4.70009671, 15.14475607, 25.58941543, 36.03407478, 46.47873414],
                        [56.9233935, 67.36805285, 77.81271221, 88.25737157, 98.70203093],
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

        def test_ew_con_var_spatial(self):
            self.assertTrue(
                (
                    test_spatial_calcs.get_calc('ew_con_var')
                    == np.array(
                        [
                            [100.0, 100.0, 100.0, 100.0, 1600.0],
                            [100.0, 100.0, 100.0, 100.0, 1600.0],
                            [100.0, 100.0, 100.0, 100.0, 1600.0],
                            [100.0, 100.0, 100.0, 100.0, 1600.0],
                        ]
                    )
                ).all()
            )

    def test_mean_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('mean'),
                np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
                rtol=1e-09,
            ).all()
        )

    def test_mean_abs_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('mean_abs'),
                np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0]),
                rtol=1e-09,
            ).all()
        )

    def test_mean_squared_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('mean_squared'),
                np.array([25.0, 16.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0]),
                rtol=1e-09,
            ).all()
        )

    def test_max_abs_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('max_abs'),
                np.array([100.0, 99.0, 98.0, 97.0, 96.0, 95.0, 96.0, 97.0, 98.0, 99.0]),
                rtol=1e-09,
            ).all()
        )

    def test_max_val_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('max_val'),
                np.array([90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0]),
                rtol=1e-09,
            ).all()
        )

    def test_min_abs_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('min_abs'),
                np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
                rtol=1e-09,
            ).all()
        )

    def test_min_val_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('min_val'),
                np.array([-100.0, -99.0, -98.0, -97.0, -96.0, -95.0, -94.0, -93.0, -92.0, -91.0]),
                rtol=1e-09,
            ).all()
        )

        def test_ns_con_var_time_series(self):
            self.assertTrue(
                np.isclose(
                    test_time_series_calcs.get_calc('ns_con_var'),
                    np.array(
                        [
                            2500.0,
                            2500.0,
                            2500.0,
                            2500.0,
                            2500.0,
                            2500.0,
                            2500.0,
                            2500.0,
                            2500.0,
                            2500.0,
                        ]
                    ),
                    rtol=1e-09,
                ).all()
            )

    def test_odds_positive_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('odds_positive'),
                np.array([0.81818182, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
                rtol=1e-09,
            ).all()
        )

    def test_prob_negative_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('prob_negative'),
                np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                rtol=1e-09,
            ).all()
        )

    def test_prob_positive_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('prob_positive'),
                np.array([0.45, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
                rtol=1e-09,
            ).all()
        )

    def test_median_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('quantile', 0.5),
                np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
                rtol=1e-09,
            ).all()
        )

    def test_rms_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('rms'),
                np.array(
                    [
                        57.87918451,
                        57.80138407,
                        57.74080013,
                        57.69748695,
                        57.67148342,
                        57.66281297,
                        57.67148342,
                        57.69748695,
                        57.74080013,
                        57.80138407,
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_std_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('std'),
                np.array(
                    [
                        59.16079783,
                        59.16079783,
                        59.16079783,
                        59.16079783,
                        59.16079783,
                        59.16079783,
                        59.16079783,
                        59.16079783,
                        59.16079783,
                        59.16079783,
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_sum_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('sum'),
                np.array([-100.0, -80.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 80.0]),
                rtol=1e-09,
            ).all()
        )

    def test_variance_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('variance'),
                np.array(
                    [
                        3325.0,
                        3325.0,
                        3325.0,
                        3325.0,
                        3325.0,
                        3325.0,
                        3325.0,
                        3325.0,
                        3325.0,
                        3325.0,
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

    def test_zscore_time_series(self):
        self.assertTrue(
            np.isclose(
                test_time_series_calcs.get_calc('zscore'),
                np.array(
                    [
                        -0.26726124,
                        -0.21380899,
                        -0.16035675,
                        -0.1069045,
                        -0.05345225,
                        0.0,
                        0.05345225,
                        0.1069045,
                        0.16035675,
                        0.21380899,
                    ]
                ),
                rtol=1e-09,
            ).all()
        )

        def test_ew_con_var_time_series(self):
            self.assertTrue(
                np.isclose(
                    test_time_series_calcs.get_calc('ew_con_var'),
                    np.array(
                        [400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]
                    ),
                    rtol=1e-09,
                ).all()
            )

    def test_diff_pcc(self):
        self.assertTrue(
            np.isclose(
                test_diff_calcs.get_diff_calc('pearson_correlation_coefficient'),
                1.0,
                rtol=1e-09,
            ).all()
        )

    def test_diff_ksp(self):
        self.assertTrue(
            np.isclose(
                test_diff_calcs.get_diff_calc('ks_p_value'),
                1.0,
                rtol=1e-09,
            ).all()
        )

    def test_diff_covariance(self):
        self.assertTrue(
            np.isclose(
                test_diff_calcs.get_diff_calc('covariance'),
                3333.25,
                rtol=1e-09,
            ).all()
        )

    def test_diff_normalized_max_pointwise_error(self):
        self.assertTrue(
            np.isclose(
                test_diff_calcs.get_diff_calc('n_emax'),
                np.array(0.00502513),
                rtol=1e-09,
            ).all()
        )

    def test_diff_normalized_root_mean_squared(self):
        self.assertTrue(
            np.isclose(
                test_diff_calcs.get_diff_calc('n_rms'),
                np.array(0.00502513),
                rtol=1e-09,
            ).all()
        )
