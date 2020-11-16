import collections

import numpy as np
import pandas as pd
import scipy.stats
import xarray as xr

from .dataarray_metrics import MetricsAccessor


class DiffMetrics:
    def __init__(self, a: xr.DataArray, b: xr.DataArray, **kwargs):
        self._metrics1 = a.ldc(**kwargs)
        self._metrics2 = b.ldc(**kwargs)
        self._metrics = pd.Series(
            collections.OrderedDict.fromkeys(
                sorted(
                    [
                        'ks_p_value',
                        'pearson_correlation_coefficient',
                        'normalized_max_pointwise_error',
                        'normalized_root_mean_squared',
                        'spatial_rel_error',
                        'ssim_value',
                        'covariance',
                    ]
                )
            )
        )
        self._is_computed = False

    def _compute_metrics(self):
        self._metrics.covariance = (
            (self._metrics2._obj - self._metrics2.metrics.mean_)
            * (self._metrics1._obj - self._metrics1.metrics.mean_)
        ).mean()
        self._metrics.ks_p_value = scipy.stats.ks_2samp(
            np.ravel(self._metrics2._obj), np.ravel(self._metrics1._obj)
        )[1]
        self._metrics.pearson_correlation_coefficient = (
            self._metrics.covariance
            / self._metrics1.metrics.standard_deviation
            / self._metrics2.metrics.standard_deviation
        )
        self._metrics.normalized_max_pointwise_error = (
            abs(self._metrics1._obj - self._metrics2._obj.max()) / self._metrics1.metrics.dyn_range
        )
        self._metrics.normalized_root_mean_square = (
            np.sqrt(np.square(self._metrics1._obj - self._metrics2.metrics.mean_))
            / self._metrics1.metrics.dyn_range
        )
        self._is_computed = True

    @property
    def metrics(self):
        if not self._is_computed:
            self._compute_metrics()
        return self._metrics
