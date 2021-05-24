import pytest
import xarray as xr

import ldcpy
from ldcpy.util import subset_data

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
ds2 = ldcpy.open_datasets(
    'cam-fv',
    ['PRECT'],
    [
        'data/cam-fv/orig.PRECT.60days.nc',
        'data/cam-fv/zfp1e-7.PRECT.60days.nc',
        'data/cam-fv/zfp1e-11.PRECT.60days.nc',
    ],
    ['orig', 'recon', 'recon_2'],
)
ds3 = ldcpy.open_datasets('cam-fv', ['T'], ['data/cam-fv/cam-fv.T.3months.nc'], ['orig'])
air_temp = xr.tutorial.open_dataset('air_temperature')


@pytest.mark.parametrize(
    'ds, varname, sets, calcs_kwargs',
    [
        (ds.isel(time=0), 'TS', ['orig', 'recon'], {'aggregate_dims': ['lat', 'lon']}),
        (ds3.isel(time=0, lev=0), 'T', ['orig', 'orig'], {'aggregate_dims': ['lat', 'lon']}),
    ],
)
def test_compare_stats(ds, varname, sets, calcs_kwargs):
    ldcpy.compare_stats(ds, varname, sets, **calcs_kwargs)


@pytest.mark.parametrize(
    'ds, kwargs',
    [(air_temp, {'subset': 'winter', 'lat': 10}), (ds3, {'lev': 10, 'lat': 10, 'lon': 20})],
)
def test_subset_data(ds, kwargs):
    s = subset_data(ds, **kwargs)
    assert isinstance(s, xr.Dataset)
