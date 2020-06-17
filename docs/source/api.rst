============
API Reference
============

This page provides an auto-generated summary of ldcpy’s API.
For more details and examples, refer to the relevant chapters in the main part of the documentation.

ldcpy Util (ldcpy.util)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: ldcpy.util

.. autofunction:: open_datasets

.. autofunction:: print_stats

ldcpy Plot (ldcpy.plot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: ldcpy.plot

.. autofunction:: plot

ldcpy Metrics (ldcpy.metrics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: ldcpy.metrics

.. autoclass:: DatasetMetrics

   .. automethod:: get_full_metric

   .. autoproperty:: ns_con_var_full
   .. autoproperty:: ew_con_var_full
   .. autoproperty:: is_positive_full
   .. autoproperty:: is_negative_full

.. autoclass:: AggregateMetrics

   .. automethod:: get_metric

   .. autoproperty:: ns_con_var
   .. autoproperty:: ew_con_var
   .. autoproperty:: mean
   .. autoproperty:: std
   .. autoproperty:: mean
   .. autoproperty:: odds_positive
   .. autoproperty:: zscore
   .. autoproperty:: mean_abs
   .. autoproperty:: prob_positive
   .. autoproperty:: prob_negative

.. autoclass:: OverallMetrics

   .. automethod:: get_overall_metric

   .. autoproperty:: zscore_cutoff
   .. autoproperty:: zscore_percent_significant


.. module:: ldcpy.error_metrics

.. autoclass:: ErrorMetrics

   .. automethod:: get_metrics_by_name


   .. autoproperty:: mean_observed
   .. autoproperty:: variance_observed
   .. autoproperty:: standard_deviation_observed
   .. autoproperty:: mean_modelled
   .. autoproperty:: variance_modelled
   .. autoproperty:: standard_deviation_modelled
   .. autoproperty:: error
   .. autoproperty:: mean_error
   .. autoproperty:: min_error
   .. autoproperty:: max_error
   .. autoproperty:: absolute_error
   .. autoproperty:: squared_error
   .. autoproperty:: mean_absolute_error
   .. autoproperty:: mean_squared_error
   .. autoproperty:: root_mean_squared_error
   .. autoproperty:: ks_p_value
   .. autoproperty:: covariance
   .. autoproperty:: pearson_correlation_coefficient
