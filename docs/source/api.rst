============
API Reference
============

This page provides an auto-generated summary of ldcpy’s API.
For more details and examples, refer to the relevant chapters in the main part of the documentation.

ldcpy Util (ldcpy.util)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: ldcpy.util

.. autofunction:: open_datasets

ldcpy Plot (ldcpy.plot)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: ldcpy.plot

.. autofunction:: plot

Parameters:
    ds: an xarray dataset containing a variable to be plotted

    varname: the variable contained in ds that contains the following dimensions: ensemble, time, lat, lon

    ens_o: the name of the ensemble member of interest

    metric: the metric to be plotted

            currently available metrics:
                'mean': mean

                'std': standard deviation

                'ns_con_var': North-South Contrast Variance

                'ew_con_var': East-West Contrast Variance

                'mean_abs': absolute value of mean

                'prob_positive': probability of positive value

                'prob_negative': probability of negative value

                'odds_positive': odds of positive value

                'zscore': the z-score at that point under the null hypothesis that value=0

                'mae_max': the mean absolute

                'lag1': the lag1 value

                'lag1_corr': correlation of each lag1 value to the previous lag1 value


    ens_r: the name of the second ensemble member of interest (required if plot_type = "spatial_comparison", or metric_type = "diff" or "ratio")

    group_by: how to group the data for time-series plotting (NOTE: currently only grouping by time is availabe) Values:

            currently available group_by values:

                'time.dayofyear'

                'time.month'

                'time.day'

                'time.year'

    scale: scale to plot time-series data on:

                'linear'

                'log'

    metric_type:

                'raw': the unaltered metric values

                'diff': the difference between the metric values in ens_o and ens_r

                'ratio': the ratio of the metric values in (ens_r/ens_o)

                'diff_metric': the metric value computed on the difference between ens_o and ens_r

    plot_type: the desired type of plot to be created

                'spatial': a plot of the world with values at each lat and lon point (takes the mean across the time dimension)

                'spatial_comparison': two side-by-side spatial plots, one of the raw metric from ens_o and the other of the raw metric from ens_r

                'time-series': A time-series plot of the data (computed by taking the mean across the lat and lon dimensions)

                'histogram': A histogram of the time-series data

    transform: the desired data transformation

                'linear': no transformation

                'log': takes the log of the metric before plotting

    subset: the desired data subset

                'first50': the first 50 days of data
                'winter': data from the months December, January, February

    lat: the latitude point of the desired time-series plot (float)

    lon: the longitude point of the desired time-series plot (float)

    color: the desired color scheme of the spatial plot (string)

    standardized_err: whether or not to standardize the error in a plot of metric_type="diff" (bool)


ldcpy Metrics (ldcpy.metrics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. module:: ldcpy.metrics

.. autoclass:: DatasetMetrics

   .. automethod:: get_full_metric

.. autoclass:: AggregateMetrics

   .. automethod:: get_metric


.. autoclass:: OverallMetrics

   .. automethod:: get_overall_metric


.. module:: ldcpy.error_metrics

.. autoclass:: ErrorMetrics

   .. automethod:: get_metrics_by_name
