import numpy as np
import xarray as xr
from scipy import stats as stats


class ErrorMetrics(object):
    """
    This class contains overall metrics for the difference between the first and second ndarrays
    """

    _available_metrics_name = {
        'mean_observed',
        'variance_observed',
        'standard_deviation_observed',
        'mean_modelled',
        'variance_modelled',
        'standard_deviation_modelled',
        'error',
        'mean_error',
        'min_error',
        'max_error',
        'absolute_error',
        'squared_error',
        'mean_absolute_error',
        'mean_squared_error',
        'root_mean_squared_error',
        'ks_p_value',
        'covariance',
        'pearson_correlation_coefficient',
    }

    def __init__(self, observed: np.ndarray, modelled: np.ndarray) -> None:
        # TODO: Support also xarray
        if isinstance(observed, np.ndarray) and isinstance(modelled, np.ndarray):
            if observed.shape != modelled.shape:
                raise ValueError('both observed and modelled must have the same shape')
            self._shape = observed.shape
            self._observed = (
                observed if (observed.dtype == np.float64) else observed.astype(np.float64)
            )
            self._mean_observed = None
            self._variance_observed = None
            self._standard_deviation_observed = None

            self._modelled = (
                modelled if (observed.dtype == np.float64) else modelled.astype(np.float64)
            )
            self._mean_modelled = None
            self._variance_modelled = None
            self._standard_deviation_modelled = None

            self._error = None
            self._mean_error = None
            self._min_error = None
            self._max_error = None
            self._absolute_error = None
            self._squared_error = None
            self._mean_absolute_error = None
            self._mean_squared_error = None
            self._root_mean_squared_error = None
            self._ks_p_value = None
            self._covariance = None
            self._pearson_correlation_coefficient = None
        else:
            raise TypeError(
                f'both measured and observed must be of type numpy.ndarray. Type(observed): {str(type(observed))}, type(modelled): {str(type(modelled))}'
            )

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    @classmethod
    def get_available_metrics_name(cls):
        return cls._available_metrics_name.copy()

    def get_all_metrics(self, exclude=None) -> dict:
        exclude = set() if exclude is None else exclude
        if isinstance(exclude, set) and all(map(lambda e: isinstance(e, str), exclude)):

            output = {}
            for name in self.get_available_metrics_name():
                if name not in exclude:
                    tmp_value = self.get_metrics_by_name(name)
                    output[name] = tmp_value if (tmp_value.size == 1) else tmp_value.tolist()
            return output
        else:
            raise TypeError('exclude must be a list of string values.')

    @property
    def observed(self) -> np.ndarray:
        return self._observed.copy()

    @observed.setter
    def observed(self, value):
        pass

    @property
    def mean_observed(self) -> np.ndarray:
        """
        mean of the observed data
        """
        if not self._is_memoized('_mean_observed'):
            self._mean_observed = self.observed.mean()

        return self._mean_observed

    @mean_observed.setter
    def mean_observed(self, value):
        pass

    @property
    def variance_observed(self) -> np.ndarray:
        """
        variance of the observed data
        """
        if not self._is_memoized('_variance_observed'):
            self._variance_observed = self.observed.var()

        return self._variance_observed

    @property
    def standard_deviation_observed(self) -> np.ndarray:
        """
        standard deviation of the observed data
        """
        if not self._is_memoized('_standard_deviation_observed'):
            self._standard_deviation_observed = np.sqrt(self.variance_observed)

        return self._standard_deviation_observed

    @standard_deviation_observed.setter
    def standard_deviation_observed(self, value):
        pass

    @variance_observed.setter
    def variance_observed(self, value):
        pass

    @property
    def modelled(self) -> np.ndarray:
        return self._modelled.copy()

    @modelled.setter
    def modelled(self, value):
        pass

    @property
    def mean_modelled(self) -> np.ndarray:
        """
        mean of the modelled data
        """
        if not self._is_memoized('_mean_modelled'):
            self._mean_modelled = self.modelled.mean()

        return self._mean_modelled

    @mean_modelled.setter
    def mean_modelled(self, value):
        pass

    @property
    def variance_modelled(self) -> np.ndarray:
        """
        variance of the modelled data
        """
        if not self._is_memoized('_variance_modelled'):
            self._variance_modelled = self.modelled.var()

        return self._variance_modelled

    @variance_modelled.setter
    def variance_modelled(self, value):
        pass

    @property
    def standard_deviation_modelled(self) -> np.ndarray:
        """
        standard deviation of the modelled data
        """
        if not self._is_memoized('_standard_deviation_modelled'):
            self._standard_deviation_modelled = np.sqrt(self.variance_modelled)

        return self._standard_deviation_modelled

    @standard_deviation_modelled.setter
    def standard_deviation_modelled(self, value):
        pass

    @property
    def error(self) -> np.ndarray:
        """
        The error at each point
        """
        if not self._is_memoized('_error'):
            self._error = self.observed - self.modelled

        return self._error

    @error.setter
    def error(self, value):
        pass

    @property
    def mean_error(self) -> np.ndarray:
        """
        The mean error
        """
        if not self._is_memoized('_mean_error'):
            self._mean_error = self.error.mean()

        return self._mean_error

    @mean_error.setter
    def mean_error(self, value):
        pass

    @property
    def min_error(self) -> np.ndarray:
        """
        The minimum error
        """
        if not self._is_memoized('_min_error'):
            self._min_error = self.error.min(initial=None)

        return self._min_error

    @min_error.setter
    def min_error(self, value):
        pass

    @property
    def max_error(self) -> np.ndarray:
        """
        The maximum error
        """
        if not self._is_memoized('_max_error'):
            self._max_error = self.error.max(initial=None)

        return self._max_error

    @max_error.setter
    def max_error(self, value):
        pass

    @property
    def absolute_error(self) -> np.ndarray:
        """
        The absolute error
        """
        if not self._is_memoized('_absolute_error'):
            self._absolute_error = self.error.__abs__()

        return self._absolute_error

    @absolute_error.setter
    def absolute_error(self, value):
        pass

    @property
    def squared_error(self) -> np.ndarray:
        """
        The squared error
        """
        if not self._is_memoized('_squared_error'):
            self._squared_error = np.power(self.error, 2)

        return self._squared_error

    @squared_error.setter
    def squared_error(self, value):
        pass

    @property
    def mean_absolute_error(self) -> np.ndarray:
        """
        The mean absolute error
        """
        if not self._is_memoized('_mean_absolute_error'):
            self._mean_absolute_error = self.absolute_error.mean()

        return self._mean_absolute_error

    @mean_absolute_error.setter
    def mean_absolute_error(self, value):
        pass

    @property
    def mean_squared_error(self):
        """
        The mean squared error
        """
        if not self._is_memoized('_mean_squared_error'):
            self._mean_squared_error = self.squared_error.mean()

        return self._mean_squared_error

    @mean_squared_error.setter
    def mean_squared_error(self, value):
        pass

    @property
    def root_mean_squared_error(self) -> np.ndarray:
        """
        The RMSE
        """
        if not self._is_memoized('_root_mean_squared_error'):
            self._root_mean_squared_error = np.sqrt(self.mean_squared_error)

        return self._root_mean_squared_error

    @root_mean_squared_error.setter
    def root_mean_squared_error(self, value):
        pass

    @property
    def ks_p_value(self):
        """
        The Kolmogorov-Smirnov p-value
        """
        if not self._is_memoized('_ks_p_value'):
            self._ks_p_value = np.asanyarray(
                stats.pearsonr(np.ravel(self.observed), np.ravel(self.modelled))
            )
        return self._ks_p_value

    @ks_p_value.setter
    def ks_p_value(self, value):
        pass

    @property
    def covariance(self) -> np.ndarray:
        """
        The covariance between the two datasets
        """
        if not self._is_memoized('_covariance'):
            self._covariance = (
                (self._modelled - self.mean_modelled) * (self._observed - self.mean_observed)
            ).mean()

        return self._covariance

    @covariance.setter
    def covariance(self, value):
        pass

    @property
    def pearson_correlation_coefficient(self) -> np.ndarray:
        """
        The pearson correlation coefficient or the two datasets
        """
        if not self._is_memoized('_pearson_correlation_coefficient'):
            self._pearson_correlation_coefficient = (
                self.covariance
                / self.standard_deviation_modelled
                / self.standard_deviation_observed
            )

        return self._pearson_correlation_coefficient

    @pearson_correlation_coefficient.setter
    def pearson_correlation_coefficient(self, value):
        pass

    def get_metrics_by_name(self, name: str):
        """
        Gets a single metric on the difference between dataset

        Parameters:
        ===========
        name -- string
            the name of the metric (must be identical to a property name)

        Returns
        =======
        out -- float32
        """
        if isinstance(name, str):
            if name == 'mean_observed':
                return self.mean_observed
            if name == 'variance_observed':
                return self.variance_modelled
            if name == 'standard_deviation_observed':
                return self.standard_deviation_observed
            if name == 'mean_modelled':
                return self.mean_modelled
            if name == 'variance_modelled':
                return self.variance_modelled
            if name == 'standard_deviation_modelled':
                return self.standard_deviation_observed
            if name == 'error':
                return self.error
            if name == 'mean_error':
                return self.mean_error
            if name == 'min_error':
                return self.min_error
            if name == 'max_error':
                return self.max_error
            if name == 'absolute_error':
                return self.absolute_error
            if name == 'squared_error':
                return self.squared_error
            if name == 'mean_absolute_error':
                return self.mean_absolute_error
            if name == 'mean_squared_error':
                return self.mean_squared_error
            if name == 'root_mean_squared_error':
                return self.root_mean_squared_error
            if name == 'ks_p_value':
                return self.ks_p_value
            if name == 'covariance':
                return self.covariance
            if name == 'pearson_correlation_coefficient':
                return self.pearson_correlation_coefficient

            raise ValueError(f'there is no metrics with the name: {name}.')
        else:
            raise TypeError('name must be a string.')
