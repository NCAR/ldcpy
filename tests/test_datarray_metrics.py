import numpy as np
import pytest

import ldcpy  # flake8: noqa

from .datasets import test_data

test_overall_metrics = test_data.ldc(aggregate_dims=['time', 'lat', 'lon']).metrics
test_spatial_metrics = test_data.ldc(aggregate_dims=['time']).metrics
test_time_series_metrics = test_data.ldc(aggregate_dims=['lat', 'lon']).metrics

expected_overall_values = {
    'dyn_range': 199,
    'ew_con_var': 2500.0,
    'lag1': np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    ),
    'lag1_first_difference': np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    ),
    'mae_day_max': np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0, 10.0, 10.0],
        ]
    ),
    'max_abs': 100,
    'max_val': 99,
    'mean_': -0.5,
    'mean_abs': 50.0,
    'mean_squared': 0.25,
    'min_abs': 0,
    'min_val': -100,
    'ns_con_var': 400.0,
    'odds_positive': 0.98019802,
    'pooled_variance': 3333.25,
    'pooled_variance_ratio': 1.0,
    'prob_negative': 0.5,
    'prob_positive': 0.495,
    'quantile_val': -0.5,
    'root_mean_squared': 57.73647028,
    'standard_deviation': 57.87918451,
    'standardized_mean': 0.0,
    'sum_': -100,
    'sum_squared': 10000,
    'variance': 3333.25,
    'zscore': -0.02731792,
}

expected_spatial_values = {
    'mean_': np.array(
        [
            [-95.5, -85.5, -75.5, -65.5, -55.5],
            [-45.5, -35.5, -25.5, -15.5, -5.5],
            [4.5, 14.5, 24.5, 34.5, 44.5],
            [54.5, 64.5, 74.5, 84.5, 94.5],
        ]
    ),
    'mean_abs': np.array(
        [
            [95.5, 85.5, 75.5, 65.5, 55.5],
            [45.5, 35.5, 25.5, 15.5, 5.5],
            [4.5, 14.5, 24.5, 34.5, 44.5],
            [54.5, 64.5, 74.5, 84.5, 94.5],
        ]
    ),
    'mean_squared': np.array(
        [
            [9120.25, 7310.25, 5700.25, 4290.25, 3080.25],
            [2070.25, 1260.25, 650.25, 240.25, 30.25],
            [20.25, 210.25, 600.25, 1190.25, 1980.25],
            [2970.25, 4160.25, 5550.25, 7140.25, 8930.25],
        ]
    ),
    'min_abs': np.array(
        [
            [91.0, 81.0, 71.0, 61.0, 51.0],
            [41.0, 31.0, 21.0, 11.0, 1.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0, 90.0],
        ]
    ),
    'max_abs': np.array(
        [
            [100.0, 90.0, 80.0, 70.0, 60.0],
            [50.0, 40.0, 30.0, 20.0, 10.0],
            [9.0, 19.0, 29.0, 39.0, 49.0],
            [59.0, 69.0, 79.0, 89.0, 99.0],
        ]
    ),
    'min_val': np.array(
        [
            [-100.0, -90.0, -80.0, -70.0, -60.0],
            [-50.0, -40.0, -30.0, -20.0, -10.0],
            [0.0, 10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0, 90.0],
        ]
    ),
    'max_val': np.array(
        [
            [-91.0, -81.0, -71.0, -61.0, -51.0],
            [-41.0, -31.0, -21.0, -11.0, -1.0],
            [9.0, 19.0, 29.0, 39.0, 49.0],
            [59.0, 69.0, 79.0, 89.0, 99.0],
        ]
    ),
    'odds_positive': np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [9.0, np.inf, np.inf, np.inf, np.inf],
            [np.inf, np.inf, np.inf, np.inf, np.inf],
        ]
    ),
    'prob_negative': np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ),
    'quantile_val': np.array(
        [
            [-95.5, -85.5, -75.5, -65.5, -55.5],
            [-45.5, -35.5, -25.5, -15.5, -5.5],
            [4.5, 14.5, 24.5, 34.5, 44.5],
            [54.5, 64.5, 74.5, 84.5, 94.5],
        ]
    ),
    'root_mean_squared': np.array(
        [
            [95.54318395, 85.54823201, 75.55461601, 65.56294685, 55.57427462],
            [45.5905692, 35.61600764, 25.66125484, 15.76388277, 6.20483682],
            [5.33853913, 14.7817455, 24.66779277, 34.61935875, 44.59260028],
            [54.57563559, 64.56392181, 74.55534857, 84.54880248, 94.54364072],
        ]
    ),
    'standard_deviation': np.array(
        [
            [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
            [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
            [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
            [3.02765035, 3.02765035, 3.02765035, 3.02765035, 3.02765035],
        ]
    ),
    'sum_': np.array(
        [
            [-955.0, -855.0, -755.0, -655.0, -555.0],
            [-455.0, -355.0, -255.0, -155.0, -55.0],
            [45.0, 145.0, 245.0, 345.0, 445.0],
            [545.0, 645.0, 745.0, 845.0, 945.0],
        ]
    ),
    'variance': np.array(
        [
            [8.25, 8.25, 8.25, 8.25, 8.25],
            [8.25, 8.25, 8.25, 8.25, 8.25],
            [8.25, 8.25, 8.25, 8.25, 8.25],
            [8.25, 8.25, 8.25, 8.25, 8.25],
        ]
    ),
    'zscore': np.array(
        [
            [-99.74649686, -89.30183751, -78.85717815, -68.41251879, -57.96785943],
            [-47.52320008, -37.07854072, -26.63388136, -16.189222, -5.74456265],
            [4.70009671, 15.14475607, 25.58941543, 36.03407478, 46.47873414],
            [56.9233935, 67.36805285, 77.81271221, 88.25737157, 98.70203093],
        ]
    ),
}

expected_values = {'overall': expected_overall_values, 'spatial': expected_spatial_values}
metrics = {'overall': test_overall_metrics, 'spatial': test_spatial_metrics}


@pytest.mark.parametrize(
    'metric',
    sorted(
        {
            'mean_',
            'mean_abs',
            'mean_squared',
            'min_abs',
            'max_abs',
            'min_val',
            'max_val',
            'odds_positive',
            'prob_negative',
            'quantile_val',
            'root_mean_squared',
            'standard_deviation',
            'sum_',
            'variance',
            'zscore',
        }
    ),
)
@pytest.mark.parametrize('k', ['overall', 'spatial'])
def test_metrics(metric, k):
    np.testing.assert_allclose(metrics[k][metric], expected_values[k][metric], rtol=1e-8)
