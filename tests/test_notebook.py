from unittest import TestCase

import numpy as np

import ldcpy
import ldcpy.plot as lp
from ldcpy.error_metrics import ErrorMetrics

ds = ldcpy.open_datasets(
    [
        '../data/cam-fv/orig.TS.100days.nc',
        '../data/cam-fv/zfp1.0.TS.100days.nc',
        '../data/cam-fv/zfp1e-1.TS.100days.nc',
    ],
    ['orig', 'recon', 'recon2'],
)
ds2 = ldcpy.open_datasets(
    [
        '../data/cam-fv/orig.PRECT.100days.nc',
        '../data/cam-fv/zfp1e-7.PRECT.100days.nc',
        '../data/cam-fv/zfp1e-11.PRECT.100days.nc',
    ],
    ['orig', 'recon', 'recon_2'],
)
ds3 = ldcpy.open_datasets(['../data/cam-fv/cam-fv.T.6months.nc'], ['orig'])


class TestNotebook(TestCase):
    def testy_test(self):
        lp.plot(ds, 'TS', 'orig', metric='quantile', quantile=0.5)
        self.assertTrue(True)
