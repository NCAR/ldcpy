import collections

import cf_xarray as cf
import dask
import numpy as np
import xarray as xr

from .metrics import DatasetMetrics, DiffMetrics


def collect_datasets(varnames, list_of_ds, labels, **kwargs):
    """
    Concatonate several different xarray datasets across a new
    "collection" dimension, which can be accessed with the specified
    labels.  Stores them in an xarray dataset which can be passed to
    the ldcpy plot functions (Call this OR open_datasets() before
    plotting.)


    Parameters
    ==========
    varnames : list
        The variable(s) of interest to combine across input files (usually just one)
    list_of_datasets : list
        The datasets to be concatonated into a collection
    labels : list
        The respective label to access data from each dataset (also used in plotting fcns)

        **kwargs :
        (optional) – Additional arguments passed on to xarray.concat(). A list of available arguments can
        be found here: https://xarray-test.readthedocs.io/en/latest/generated/xarray.concat.html

    Returns
    =======
    out : xarray.Dataset
          a collection containing all the data from the list datasets

    """
    # Error checking:
    # list_of_files and labels must be same length
    assert len(list_of_ds) == len(
        labels
    ), 'ERROR:collect_dataset dataset list and labels arguments must be the same length'

    # the number of timeslices must be the same
    sz = np.zeros(len(list_of_ds))
    for i, myds in enumerate(list_of_ds):
        sz[i] = myds.sizes['time']
    indx = np.unique(sz)
    assert indx.size == 1, 'ERROR: all datasets must have the same length time dimension'

    # preprocess
    for i, myds in enumerate(list_of_ds):
        list_of_ds[i] = preprocess(myds, varnames)

    full_ds = xr.concat(list_of_ds, 'collection', **kwargs)

    full_ds['collection'] = xr.DataArray(labels, dims='collection')

    print('dataset size in GB {:0.2f}\n'.format(full_ds.nbytes / 1e9))

    return full_ds


def open_datasets(varnames, list_of_files, labels, **kwargs):
    """
    Open several different netCDF files, concatenate across
    a new 'collection' dimension, which can be accessed with the specified
    labels. Stores them in an xarray dataset which can be passed to the ldcpy
    plot functions.

    Parameters
    ==========
    varnames : list
           The variable(s) of interest to combine across input files (usually just one)
    list_of_files : list
        The file paths for the netCDF file(s) to be opened
    labels : list
        The respective label to access data from each netCDF file (also used in plotting fcns)

    **kwargs :
        (optional) – Additional arguments passed on to xarray.open_mfdataset(). A list of available arguments can
        be found here: http://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html

    Returns
    =======
    out : xarray.Dataset
          a collection containing all the data from the list of files


    """

    # Error checking:
    # list_of_files and labels must be same length
    assert len(list_of_files) == len(
        labels
    ), 'ERROR: open_dataset file list and labels arguments must be the same length'

    # all must have the same time dimension
    sz = np.zeros(len(list_of_files))
    for i, myfile in enumerate(list_of_files):
        myds = xr.open_dataset(myfile)
        sz[i] = myds.sizes['time']
        myds.close()
    indx = np.unique(sz)
    assert indx.size == 1, 'ERROR: all files must have the same length time dimension'

    # preprocess_vars is here for working on jupyter hub...
    def preprocess_vars(ds):
        return ds[varnames]

    full_ds = xr.open_mfdataset(
        list_of_files,
        concat_dim='collection',
        combine='nested',
        data_vars=varnames,
        parallel=True,
        preprocess=preprocess_vars,
        **kwargs,
    )

    full_ds['collection'] = xr.DataArray(labels, dims='collection')
    print('dataset size in GB {:0.2f}\n'.format(full_ds.nbytes / 1e9))

    return full_ds


def preprocess(ds, varnames):
    return ds[varnames]


def compare_stats(
    ds,
    varname: str,
    set1: str,
    set2: str,
    significant_digits: int = 5,
    include_ssim_metric: bool = False,
    **metrics_kwargs,
):
    """
    Print error summary statistics of two DataArrays

    Parameters
    ==========
    ds : xarray.Dataset
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname : str
        The variable of interest in the dataset
    set1 : str
        The collection label of the "control" data
    set2 : str
        The collection label of the (1st) data to compare
    significant_digits : int, optional
        The number of significant digits to use when printing stats, (default 5)
    include_ssim_metric : bool, optional
        Whether or not to compute the ssim metric, (default: False)
    **metrics_kwargs :
        Additional keyword arguments passed through to the
        :py:class:`~ldcpy.DatasetMetrics` instance.

    Returns
    =======
    out : None

    """

    # get a datarray for the variable of interest and get collections
    # (this is done seperately to work with cf_xarray)

    if varname == 'T':  # work around for cf_xarray (until new tag that
        # includes issue 130 updated to main on 1/27/21)
        ds.T.attrs['standard_name'] = 'tt'
        da = ds.cf['tt']
    else:
        da = ds.cf[varname]

    # use this after the update instead of above
    # da = ds.cf.data_vars[varname]

    da1 = da.sel(collection=set1)
    da2 = da.sel(collection=set2)
    dd = da1 - da2

    aggregate_dims = metrics_kwargs.pop('aggregate_dims', None)

    ds0_metrics = DatasetMetrics(da1, aggregate_dims, **metrics_kwargs)

    ds1_metrics = DatasetMetrics(da2, aggregate_dims, **metrics_kwargs)

    d_metrics = DatasetMetrics(
        dd,
        aggregate_dims,
        **metrics_kwargs,
    )

    diff_metrics = DiffMetrics(
        da1,
        da2,
        aggregate_dims,
        **metrics_kwargs,
    )

    output = collections.OrderedDict()
    output['skip1'] = 0
    output[f'mean {set1}'] = ds0_metrics.get_metric('mean').data
    output[f'mean {set2}'] = ds1_metrics.get_metric('mean').data
    output['mean diff'] = d_metrics.get_metric('mean').data
    output['skip2'] = 0
    output[f'variance {set1}'] = ds0_metrics.get_metric('variance').data
    output[f'variance {set2}'] = ds1_metrics.get_metric('variance').data
    output['skip3'] = 0
    output[f'standard deviation {set1}'] = ds0_metrics.get_metric('std').data
    output[f'standard deviation {set2}'] = ds1_metrics.get_metric('std').data
    output['skip4'] = 0
    output[f'max value {set1}'] = ds0_metrics.get_metric('max_val').data
    output[f'max value {set2}'] = ds1_metrics.get_metric('max_val').data
    output[f'min value {set1}'] = ds0_metrics.get_metric('min_val').data
    output[f'min value {set2}'] = ds1_metrics.get_metric('min_val').data
    output['skip55'] = 0
    output['max abs diff'] = d_metrics.get_metric('max_abs').data
    output['min abs diff'] = d_metrics.get_metric('min_abs').data
    output['mean abs diff'] = d_metrics.get_metric('mean_abs').data
    output['mean squared diff'] = d_metrics.get_metric('mean_squared').data
    output['root mean squared diff'] = d_metrics.get_metric('rms').data
    output['normalized root mean squared diff'] = diff_metrics.get_diff_metric('n_rms').data
    output['normalized max pointwise error'] = diff_metrics.get_diff_metric('n_emax').data
    output['pearson correlation coefficient'] = diff_metrics.get_diff_metric(
        'pearson_correlation_coefficient'
    ).data
    output['ks p-value'] = diff_metrics.get_diff_metric('ks_p_value')
    tmp = 'spatial relative error(% > ' + str(ds0_metrics.get_single_metric('spre_tol')) + ')'
    output[tmp] = diff_metrics.get_diff_metric('spatial_rel_error')
    output['max spatial relative error'] = diff_metrics.get_diff_metric('max_spatial_rel_error')

    if include_ssim_metric:
        output['ssim'] = diff_metrics.get_diff_metric('ssim')
        output['ssim_fp'] = diff_metrics.get_diff_metric('ssim_fp')
        # output['ssim_fp_old'] = diff_metrics.get_diff_metric('ssim_fp_old')

    if dask.is_dask_collection(ds):
        output = dask.compute(output)[0]

    for key, value in output.items():
        if key[:4] != 'skip':
            # rounded_value = f'{float(f"{value:.{significant_digits}g}"):g}'
            rounded_value = f'{(f"{value:.{significant_digits}g}")}'

            print(f'{key:<35}:', f'{rounded_value}')
        else:
            print(' ')


def check_metrics(
    ds,
    varname,
    set1,
    set2,
    ks_tol=0.05,
    pcc_tol=0.99999,
    spre_tol=5.0,
    ssim_tol=0.99995,
    **metrics_kwargs,
):
    """

    Check the K-S, Pearson Correlation, and Spatial Relative Error metrics

    Parameters
    ==========
    ds : xarray.Dataset
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname : str
        The variable of interest in the dataset
    set1 : str
        The collection label of the "control" data
    set2 : str
        The collection label of the (1st) data to compare
    ks_tol : float, optional
        The p-value threshold (significance level) for the K-S test (default = .05)
    pcc_tol: float, optional
        The default Pearson corrolation coefficient (default  = .99999)
    spre_tol: float, optional
        The percentage threshold for failing grid points in the spatial relative error test (default = 5.0).
    ssim_tol: float, optional
         The threshold for the ssim test (default = .999950
    **metrics_kwargs :
        Additional keyword arguments passed through to the
        :py:class:`~ldcpy.DatasetMetrics` instance.

    Returns
    =======
    out : Number of failing metrics

    Notes
    ======

    Check the K-S, Pearson Correlation, and Spatial Relative Error metrics from:

    A. H. Baker, H. Xu, D. M. Hammerling, S. Li, and J. Clyne,
    “Toward a Multi-method Approach: Lossy Data Compression for
    Climate Simulation Data”, in J.M. Kunkel et al. (Eds.): ISC
    High Performance Workshops 2017, Lecture Notes in Computer
    Science 10524, pp. 30–42, 2017 (doi:10.1007/978-3-319-67630-2_3).

    Check the SSIM metric from:

    A.H. Baker, D.M. Hammerling, and T.L. Turton. “Evaluating image
    quality measures to assess the impact of lossy data compression
    applied to climate simulation data”, Computer Graphics Forum 38(3),
    June 2019, pp. 517-528 (doi:10.1111/cgf.13707).

    Default tolerances for the tests are:
    ------------------------
    K-S: fail if p-value < .05 (significance level)
    Pearson correlation coefficient:  fail if coefficient < .99999
    Spatial relative error: fail if > 5% of grid points fail relative error
    SSIM: fail if SSIM < .99995

    """

    print(f'Evaluating 4 metrics for {set1} data (set1) and {set2} data (set2):')
    aggregate_dims = metrics_kwargs.pop('aggregate_dims', None)
    diff_metrics = DiffMetrics(
        ds[varname].sel(collection=set1),
        ds[varname].sel(collection=set2),
        aggregate_dims,
        **metrics_kwargs,
    )

    # count the number of failures
    num_fail = 0
    # Pearson less than pcc_tol means fail
    pcc = diff_metrics.get_diff_metric('pearson_correlation_coefficient').data.compute()
    if pcc < pcc_tol:
        print('     *FAILED pearson correlation coefficient test...(pcc = {0:.5f}'.format(pcc), ')')
        num_fail = num_fail + 1
    else:
        print('     PASSED pearson correlation coefficient test...(pcc = {0:.5f}'.format(pcc), ')')
    # K-S p-value less than ks_tol means fail (can reject null hypo)
    ks = diff_metrics.get_diff_metric('ks_p_value')
    if ks < ks_tol:
        print('     *FAILED ks test...(ks p_val = {0:.4f}'.format(ks), ')')
        num_fail = num_fail + 1
    else:
        print('     PASSED ks test...(ks p_val = {0:.4f}'.format(ks), ')')
    # Spatial rel error fails if more than spre_tol
    spre = diff_metrics.get_diff_metric('spatial_rel_error')
    if spre > spre_tol:
        print('     *FAILED spatial relative error test ... (spre = {0:.2f}'.format(spre), ' %)')
        num_fail = num_fail + 1
    else:
        print('     PASSED spatial relative error test ...(spre = {0:.2f}'.format(spre), ' %)')
    # SSIM less than of ssim_tol is failing
    ssim_val = diff_metrics.get_diff_metric('ssim')
    if ssim_val < ssim_tol:
        print('     *FAILED SSIM test ... (ssim = {0:.5f}'.format(ssim_val), ')')
        num_fail = num_fail + 1
    else:
        print('     PASSED SSIM test ... (ssim = {0:.5f}'.format(ssim_val), ')')
    if num_fail > 0:
        print(f'WARNING: {num_fail} of 4 tests failed.')
    return num_fail


def subset_data(
    ds,
    subset=None,
    lat=None,
    lon=None,
    lev=None,
    start=None,
    end=None,
    time_dim_name='time',
    vertical_dim_name=None,
    lat_coord_name=None,
    lon_coord_name=None,
):
    """
    Get a subset of the given dataArray, returns a dataArray
    """
    ds_subset = ds

    # print(ds.cf.describe())

    if lon_coord_name is None:
        lon_coord_name = ds.cf.coordinates['longitude'][0]
    if lat_coord_name is None:
        lat_coord_name = ds.cf.coordinates['latitude'][0]
    if vertical_dim_name is None:
        try:
            vert = ds.cf['vertical']
        except KeyError:
            vert = None
        if vert is not None:
            vertical_dim_name = ds.cf.coordinates['vertical'][0]

    # print(lat_coord_name, lon_coord_name, vertical_dim_name)

    latdim = ds_subset.cf[lon_coord_name].ndim
    # need dim names
    dd = ds_subset.cf['latitude'].dims
    if latdim == 1:
        lat_dim_name = dd[0]
        lon_dim_name = ds_subset.cf['longitude'].dims[0]
    elif latdim == 2:
        lat_dim_name = dd[0]
        lon_dim_name = dd[1]

    if start is not None and end is not None:
        ds_subset = ds_subset.isel({time_dim_name: slice(start, end + 1)})

    if subset is not None:
        if subset == 'winter':
            ds_subset = ds_subset.cf.sel(time=ds.cf['time'].dt.season == 'DJF')
        elif subset == 'spring':
            ds_subset = ds_subset.cf.sel(time=ds.cf['time'].dt.season == 'MAM')
        elif subset == 'summer':
            ds_subset = ds_subset.cf.sel(time=ds.cf['time'].dt.season == 'JJA')
        elif subset == 'autumn':
            ds_subset = ds_subset.cf.sel(time=ds.cf['time'].dt.season == 'SON')

        elif subset == 'first5':
            ds_subset = ds_subset.isel({time_dim_name: slice(None, 5)})

    if lev is not None:
        if vertical_dim_name in ds_subset.dims:
            ds_subset = ds_subset.isel({vertical_dim_name: lev})

    if latdim == 1:

        if lat is not None:
            ds_subset = ds_subset.sel(**{lat_coord_name: [lat], 'method': 'nearest'})
        if lon is not None:
            ds_subset = ds_subset.sel(**{lon_coord_name: [lon + 180], 'method': 'nearest'})

    elif latdim == 2:

        # print(ds_subset)

        if lat is not None:
            if lon is not None:

                # lat is -90 to 90
                # lon should be 0- 360
                ad_lon = lon
                if ad_lon < 0:
                    ad_lon = ad_lon + 360

                mlat = ds_subset[lat_coord_name].compute()
                mlon = ds_subset[lon_coord_name].compute()
                # euclidean dist for now....
                di = np.sqrt(np.square(ad_lon - mlon) + np.square(lat - mlat))
                index = np.where(di == np.min(di))
                xmin = index[0][0]
                ymin = index[1][0]

                # Don't want if it's a land point
                check = ds_subset.isel(nlat=xmin, nlon=ymin, time=1).compute()
                if np.isnan(check):
                    print(
                        'You have chosen a lat/lon point with Nan values (i.e., a land point). Plot will not make sense.'
                    )
                ds_subset = ds_subset.isel({lat_dim_name: [xmin], lon_dim_name: [ymin]})

                # ds_subset.compute()

    return ds_subset
