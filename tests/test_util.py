from unittest import TestCase

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
class TestUtil(TestCase):
    @pytest.mark.parametrize(
        'ds, kwargs',
        [(air_temp, {'subset': 'winter', 'lat': 10}), (ds3, {'lev': 10, 'lat': 10, 'lon': 20})],
    )
    def test_subset_data(self, ds, kwargs):
        s = subset_data(ds, **kwargs)
        assert isinstance(s, xr.Dataset)

    def test_open_datasets(self):
        dataDir = '/Users/alex/Desktop/data'

        # Original file name:
        # b.e11.B20TRC5CNBDRD.f09_g16.030.cam.h1.{daily_variable}.19200101-20051231.nc
        cols_daily = {}
        for daily_variable in ['TS']:
            cols_daily[daily_variable] = ldcpy.open_datasets(
                'cam-fv',
                [daily_variable],
                [
                    f'{dataDir}/orig.{daily_variable.lower()}.100.nc',
                    f'{dataDir}/0.0001.{daily_variable.lower()}.100.nc',
                    f'{dataDir}/0.001.{daily_variable.lower()}.100.nc',
                    f'{dataDir}/0.01.{daily_variable.lower()}.100.nc',
                    f'{dataDir}/0.1.{daily_variable.lower()}.100.nc',
                ],
                ['orig', '0001', '001', '01', '1'],
            )
        self.assertTrue(True)
