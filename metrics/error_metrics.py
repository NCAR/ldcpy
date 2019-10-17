import numpy as np
from scipy import stats as stats


class ErrorMetrics(object):
    def __init__(self, observed: np.ndarray, modelled: np.ndarray) -> None:
        if isinstance(observed, np.ndarray) and isinstance(modelled, np.ndarray):
            self._observed = observed if (observed.dtype == np.float64) else observed.astype(np.float64)
            self._mean_observed = None
            self._variance_observed = None
            self._standard_deviation_observed = None

            self._modelled = modelled if (observed.dtype == np.float64) else modelled.astype(np.float64)
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
            raise TypeError("both measured and observed must be of type numpy.ndarray")

    def _is_memoized(self, metric_name: str) -> bool:
        return hasattr(self, metric_name) and (self.__getattribute__(metric_name) is not None)

    def get_all_metrics(self) -> dict:
        return {
            "mean_observed":                    self.mean_observed,
            "variance_observed":                self.variance_observed,
            "standard_deviation_observed":      self.standard_deviation_observed,

            "mean_modelled":                    self.mean_modelled,
            "variance_modelled":                self.variance_modelled,
            "standard_deviation_modelled":      self.standard_deviation_modelled,

            "error":                            self.error.tolist(),
            "mean_error":                       self.mean_error,
            "min_error":                        self.min_error,
            "max_error":                        self.max_error,
            "absolute_error":                   self.absolute_error.tolist(),
            "squared_error":                    self.squared_error.tolist(),
            "mean_absolute_error":              self.mean_absolute_error,
            "mean_squared_erro":                self.mean_squared_error,
            "root_mean_squared_error":          self.root_mean_squared_error,
            "ks_p_value":                       self.ks_p_value,
            "covariance":                       self.covariance,
            "pearson_correlation_coefficient":  self.pearson_correlation_coefficient
        }

    @property
    def observed(self) -> np.ndarray:
        return self._observed.copy()

    @observed.setter
    def observed(self, value):
        pass

    @property
    def mean_observed(self) -> np.ndarray:
        if not self._is_memoized("_mean_observed"):
            self._mean_observed = self.observed.mean()

        return self._mean_observed

    @mean_observed.setter
    def mean_observed(self, value):
        pass

    @property
    def variance_observed(self) -> np.ndarray:
        if not self._is_memoized("_variance_observed"):
            self._variance_observed = self.observed.var()

        return self._variance_observed

    @property
    def standard_deviation_observed(self) -> np.ndarray:
        if not self._is_memoized("_standard_deviation_observed"):
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
        if not self._is_memoized("_mean_modelled"):
            self._mean_modelled = self.modelled.mean()

        return self._mean_modelled

    @mean_modelled.setter
    def mean_modelled(self, value):
        pass

    @property
    def variance_modelled(self) -> np.ndarray:
        if not self._is_memoized("_variance_modelled"):
            self._variance_modelled = self.modelled.var()

        return self._variance_modelled

    @variance_modelled.setter
    def variance_modelled(self, value):
        pass

    @property
    def standard_deviation_modelled(self) -> np.ndarray:
        if not self._is_memoized("_standard_deviation_modelled"):
            self._standard_deviation_modelled = np.sqrt(self.variance_modelled)

        return self._standard_deviation_modelled

    @standard_deviation_modelled.setter
    def standard_deviation_modelled(self, value):
        pass

    @property
    def error(self) -> np.ndarray:
        if not self._is_memoized("_error"):
            self._error = self.observed - self.modelled

        return self._error

    @error.setter
    def error(self, value):
        pass

    @property
    def mean_error(self) -> np.ndarray:
        if not self._is_memoized("_mean_error"):
            self._mean_error = self.error.mean()

        return self._mean_error

    @mean_error.setter
    def mean_error(self, value):
        pass

    @property
    def min_error(self) -> np.ndarray:
        if not self._is_memoized("_min_error"):
            self._min_error = self.error.min(initial=None)

        return self._min_error

    @min_error.setter
    def min_error(self, value):
        pass

    @property
    def max_error(self) -> np.ndarray:
        if not self._is_memoized("_max_error"):
            self._max_error = self.error.max(initial=None)

        return self._max_error

    @max_error.setter
    def max_error(self, value):
        pass

    @property
    def absolute_error(self) -> np.ndarray:
        if not self._is_memoized("_absolute_error"):
            self._absolute_error = self.error.__abs__()

        return self._absolute_error

    @absolute_error.setter
    def absolute_error(self, value):
        pass

    @property
    def squared_error(self) -> np.ndarray:
        if not self._is_memoized("_squared_error"):
            self._squared_error = np.power(self.error, 2)

        return self._squared_error

    @squared_error.setter
    def squared_error(self, value):
        pass

    @property
    def mean_absolute_error(self) -> np.ndarray:
        if not self._is_memoized("_mean_absolute_error"):
            self._mean_absolute_error = self.absolute_error.mean()

        return self._mean_absolute_error

    @mean_absolute_error.setter
    def mean_absolute_error(self, value):
        pass

    @property
    def mean_squared_error(self):
        if not self._is_memoized("_mean_squared_error"):
            self._mean_squared_error = self.squared_error.mean()

        return self._mean_squared_error

    @mean_squared_error.setter
    def mean_squared_error(self, value):
        pass

    @property
    def root_mean_squared_error(self) -> np.ndarray:
        if not self._is_memoized("_root_mean_squared_error"):
            self._root_mean_squared_error = np.sqrt(self.mean_squared_error)

        return self._root_mean_squared_error

    @root_mean_squared_error.setter
    def root_mean_squared_error(self, value):
        pass

    @property
    def ks_p_value(self):
        if not self._is_memoized("_ks_p_value"):
            self._ks_p_value = stats.pearsonr(
                np.ravel(self.observed),
                np.ravel(self.modelled)
            )
        return self._ks_p_value

    @ks_p_value.setter
    def ks_p_value(self, value):
        pass

    @property
    def covariance(self) -> np.ndarray:
        if not self._is_memoized("_covariance"):
            self._covariance = (
                (self._modelled - self.mean_modelled) * (self._observed - self.mean_observed)
            ).mean()

        return self._covariance

    @covariance.setter
    def covariance(self, value):
        pass

    @property
    def pearson_correlation_coefficient(self) -> np.ndarray:
        if not self._is_memoized("_pearson_correlation_coefficient"):
            self._pearson_correlation_coefficient = \
                self.covariance / self.standard_deviation_modelled / self.standard_deviation_observed

        return self._pearson_correlation_coefficient

    @pearson_correlation_coefficient.setter
    def pearson_correlation_coefficient(self, value):
        pass
