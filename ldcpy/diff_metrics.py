import collections
import copy
import tempfile

import cartopy.crs
import cartopy.util
import matplotlib
import matplotlib.pyplot
import numpy as np
import pandas as pd
import scipy.stats
import skimage.io
import skimage.metrics
import xarray as xr

from .dataarray_metrics import MetricsAccessor  # needed for the .ldc accessor to work


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

    def _spatial_rel_error(self):
        sp_tol = self._metrics1._spre_tol
        t1 = np.ravel(self._metrics1._obj)
        t2 = np.ravel(self._metrics2._obj)

        # check for zeros in t1 (if zero then change to 1 - which
        # does an absolute error at that point)
        t1 = np.where(abs(t1) == 0.0, 1.0, t1)
        m_tt = t1 - t2
        m_tt = m_tt / t1
        a = len(m_tt[m_tt > sp_tol])
        return (a / m_tt.shape[0]) * 100

    def _ssim_value(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            filename_1, filename_2 = f'{tmpdirname}/t_ssim1.png', f'{tmpdirname}/t_ssim2.png'
            d1 = self._metrics1._obj
            d2 = self._metrics2._obj
            lat1 = d1[self._metrics1._lat_dim_name]
            lat2 = d1[self._metrics2._lat_dim_name]
            cy1, lon1 = cartopy.util.add_cyclic_point(d1, coord=d1[self._metrics1._lon_dim_name])
            cy2, lon2 = cartopy.util.add_cyclic_point(d2, coord=d2[self._metrics2._lon_dim_name])

            backend_ = matplotlib.get_backend()
            matplotlib.use('Agg')

            no_inf_d1 = np.nan_to_num(cy1, nan=np.nan)
            no_inf_d2 = np.nan_to_num(cy2, nan=np.nan)

            color_min = min(
                np.min(d1.where(d1 != -np.inf)).min(), np.min(d2.where(d2 != -np.inf)).min()
            )
            color_max = max(
                np.max(d1.where(d1 != np.inf)).max(), np.max(d2.where(d2 != np.inf)).max()
            )

            fig = matplotlib.pyplot.figure(dpi=300, figsize=(9, 2.5))
            mymap = copy.copy(matplotlib.pyplot.cm.get_cmap('coolwarm'))
            mymap.set_under(color='black')
            mymap.set_over(color='white')
            mymap.set_bad(alpha=0)

            ax1 = matplotlib.pyplot.subplot(
                1, 2, 1, projection=cartopy.crs.Robinson(central_longitude=0.0)
            )
            ax1.set_facecolor('#39ff14')

            ax1.pcolormesh(
                lon1,
                lat1,
                no_inf_d1,
                transform=cartopy.crs.PlateCarree(),
                cmap=mymap,
                vmin=color_min,
                vmax=color_max,
            )
            ax1.set_global()
            ax1.coastlines(linewidth=0.5)
            ax1.axis('off')
            matplotlib.pyplot.margins(0, 0)
            extent1 = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            ax1.imshow
            matplotlib.pyplot.savefig(
                filename_1, bbox_inches=extent1, transparent=True, pad_inches=0
            )
            ax1.axis('on')

            ax2 = matplotlib.pyplot.subplot(
                1, 2, 2, projection=cartopy.crs.Robinson(central_longitude=0.0)
            )
            ax2.set_facecolor('#39ff14')

            ax2.pcolormesh(
                lon2,
                lat2,
                no_inf_d2,
                transform=cartopy.crs.PlateCarree(),
                cmap=mymap,
                vmin=color_min,
                vmax=color_max,
            )
            ax2.set_global()
            ax2.coastlines(linewidth=0.5)
            matplotlib.pyplot.margins(0, 0)
            ax2.imshow
            ax2.axis('off')
            extent2 = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            matplotlib.pyplot.savefig(
                filename_2, bbox_inches=extent2, transparent=True, pad_inches=0
            )
            ax2.axis('on')

            img1 = skimage.io.imread(filename_1)
            img2 = skimage.io.imread(filename_2)
            s = skimage.metrics.structural_similarity(img1, img2, multichannel=True)

            # Reset backend
            matplotlib.use(backend_)
            return s

    def _compute_metrics(self):
        self._metrics.covariance = (
            (self._metrics2._obj - self._metrics2.metrics.mean_val)
            * (self._metrics1._obj - self._metrics1.metrics.mean_val)
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
            np.sqrt(np.square(self._metrics1._obj - self._metrics2.metrics.mean_val))
            / self._metrics1.metrics.dyn_range
        )

        self._metrics.spatial_rel_error = self._spatial_rel_error()
        self._metrics.ssim_value = self._ssim_value()

        self._is_computed = True

    @property
    def metrics(self):
        if not self._is_computed:
            self._compute_metrics()
        return self._metrics
