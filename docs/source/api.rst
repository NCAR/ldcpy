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

   .. automethod:: get_metric
   .. automethod:: get_single_metric

   .. autoproperty:: ns_con_var
   .. autoproperty:: ew_con_var
   .. autoproperty:: std
   .. autoproperty:: mean
   .. autoproperty:: mean_abs
   .. autoproperty:: mean_squared
   .. autoproperty:: root_mean_squared
   .. autoproperty:: sum
   .. autoproperty:: sum_squared
   .. autoproperty:: odds_positive
   .. autoproperty:: zscore
   .. autoproperty:: mean_abs
   .. autoproperty:: prob_positive
   .. autoproperty:: prob_negative
   .. autoproperty:: quantile
   .. autoproperty:: zscore_cutoff
   .. autoproperty:: zscore_percent_significant

.. autoclass:: DiffMetrics

   .. automethod:: get_diff_metric

   .. autoproperty:: covariance
   .. autoproperty:: pearson_correlation_coefficient
   .. autoproperty:: ks_p_value
