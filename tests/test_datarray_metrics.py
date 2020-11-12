import numpy as np
import pandas as pd
import pytest
import xarray as xr

import ldcpy  # flake8: noqa

times = pd.date_range('2000-01-01', periods=10)
lats = [0, 1, 2, 3]
lons = [0, 1, 2, 3, 4]
test_data = xr.DataArray(
    np.arange(-100, 100).reshape(4, 5, 10),
    coords=[lats, lons, times],
    dims=['lat', 'lon', 'time'],
)
test_data_2 = xr.DataArray(
    np.arange(-99, 101).reshape(4, 5, 10),
    coords=[lats, lons, times],
    dims=['lat', 'lon', 'time'],
)

test_overall_metrics = test_data.ldc(aggregate_dims=['time', 'lat', 'lon']).metrics
expected_values = {
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
    'quantile_value': -0.5,
    'root_mean_squared': 57.73647028,
    'standard_deviation': 57.87918451,
    'standardized_mean': 0.0,
    'sum_': -100,
    'sum_squared': 10000,
    'variance': 3333.25,
    'zscore': -0.02731792,
}


@pytest.mark.parametrize('metrics, expected', [(test_overall_metrics, expected_values)])
@pytest.mark.parametrize('key', sorted(expected_values.keys()))
def test_metrics(metrics, expected, key):
    assert isinstance(metrics, pd.Series)
    np.testing.assert_allclose(metrics[key], expected[key])
