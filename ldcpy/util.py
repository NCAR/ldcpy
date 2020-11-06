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


def compare_stats(ds, varname, set1, set2, time=0, significant_digits=4):
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
    time : int, optional
        The time index used to compare the two netCDF files (default 0)

    significant_digits : int, optional
        The number of significant digits to use when printing stats, (default 4)

    Returns
    =======
    out : None

    """
    print('Comparing {} data (set1) to {} data (set2) at time = {}'.format(set1, set2, time))

    # Make sure we don't exceed time bound
    time_mx = ds[varname].sel(collection=set1).sizes['time'] - 1
    if time > time_mx:
        raise ValueError(f'specified time index exceeds max time dimension {time_mx}.')

    # check for names of lat/lon
    # CAM-FV: dims = lat, lon; coords = lat, lon
    # POP: dims = nlat, nlon; coords = ULAT,ULONG, TLAT, TLONG
    # CAM-SE: to do

    if 'lat' in list(ds.dims):
        data_type = 'cam-fv'
        print('CAM-FV data...')
        ds0_metrics = DatasetMetrics(
            ds[varname].sel(collection=set1).isel(time=time), ['lat', 'lon']
        )
        ds1_metrics = DatasetMetrics(
            ds[varname].sel(collection=set2).isel(time=time), ['lat', 'lon']
        )
        d_metrics = DatasetMetrics(
            ds[varname].sel(collection=set1).isel(time=time)
            - ds[varname].sel(collection=set2).isel(time=time),
            ['lat', 'lon'],
        )
        diff_metrics = DiffMetrics(
            ds[varname].sel(collection=set1).isel(time=time),
            ds[varname].sel(collection=set2).isel(time=time),
            ['lat', 'lon'],
        )
    elif 'nlat' in list(ds.dims):
        data_type = 'pop'
        print('POP data...')
        ds0_metrics = DatasetMetrics(
            ds[varname].sel(collection=set1).isel(time=time), ['nlat', 'nlon']
        )
        ds1_metrics = DatasetMetrics(
            ds[varname].sel(collection=set2).isel(time=time), ['nlat', 'nlon']
        )
        d_metrics = DatasetMetrics(
            ds[varname].sel(collection=set1).isel(time=time)
            - ds[varname].sel(collection=set2).isel(time=time),
            ['nlat', 'nlon'],
        )
        diff_metrics = DiffMetrics(
            ds[varname].sel(collection=set1).isel(time=time),
            ds[varname].sel(collection=set2).isel(time=time),
            ['nlat', 'nlon'],
        )
    else:
        data_type = 'unk'
        print('Type of data not recognized.')

    output = {}

    output['skip1'] = 0

    output['mean set1'] = ds0_metrics.get_metric('mean').data.compute()
    output['mean set2'] = ds1_metrics.get_metric('mean').data.compute()
    output['mean diff'] = d_metrics.get_metric('mean').data.compute()

    output['skip2'] = 0

    output['variance set1'] = ds0_metrics.get_metric('variance').data.compute()
    output['variance set2'] = ds1_metrics.get_metric('variance').data.compute()

    output['skip3'] = 0

    output['standard deviation set1'] = ds0_metrics.get_metric('std').data.compute()
    output['standard deviation set2'] = ds1_metrics.get_metric('std').data.compute()

    output['skip4'] = 0

    output['max value set1'] = ds0_metrics.get_metric('max_val').data.compute()
    output['max value set2'] = ds1_metrics.get_metric('max_val').data.compute()
    output['min value set1'] = ds0_metrics.get_metric('min_val').data.compute()
    output['min value set2'] = ds1_metrics.get_metric('min_val').data.compute()

    output['skip55'] = 0

    output['max abs diff'] = d_metrics.get_metric('max_abs').data.compute()
    output['min abs diff'] = d_metrics.get_metric('min_abs').data.compute()
    output['mean abs diff'] = d_metrics.get_metric('mean_abs').data.compute()

    output['mean squared diff'] = d_metrics.get_metric('mean_squared').data.compute()
    output['root mean squared diff'] = d_metrics.get_metric('rms').data.compute()

    output['normalized root mean squared diff'] = diff_metrics.get_diff_metric(
        'n_rms'
    ).data.compute()
    output['normalized max pointwise error'] = diff_metrics.get_diff_metric('n_emax').data.compute()
    output['pearson correlation coefficient'] = diff_metrics.get_diff_metric(
        'pearson_correlation_coefficient'
    ).data.compute()
    output['ks p-value'] = diff_metrics.get_diff_metric('ks_p_value')
    tmp = 'spatial relative error(% > ' + str(ds0_metrics.get_single_metric('spre_tol')) + ')'
    output[tmp] = diff_metrics.get_diff_metric('spatial_rel_error')
    # don't do sim for pop
    if not data_type == 'pop':
        output['ssim'] = diff_metrics.get_diff_metric('ssim')

    for key, value in output.items():
        if key[:4] != 'skip':
            rounded_value = f'{float(f"{value:.{significant_digits}g}"):g}'
            print(f'{key}: {rounded_value}')
        else:
            print(' ')


def check_metrics(
    ds, varname, set1, set2, time=0, ks_tol=0.05, pcc_tol=0.99999, spre_tol=5.0, ssim_tol=0.99995
):
    """

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
    time : int, optional
        The time index used t (default = 0)
    ks_tol : float, optional
        The p-value threshold (significance level) for the K-S test (default = .05)
    pcc_tol: float, optional
        The default Pearson corrolation coefficient (default  = .99999)
    spre_tol: float, optional
        The percentage threshold for failing grid points in the spatial relative error test (default = 5.0).
    ssim_tol: float, optional
         The threshold for the ssim test (default = .999950

    Returns
    =======
    out : Number of failing metrics

    """

    # count the number of failuress
    num_fail = 0

    print('Evaluating 4 metrics for {} data (set1) and {} data (set2)'.format(set1, set2), ':')

    diff_metrics = DiffMetrics(
        ds[varname].sel(collection=set1).isel(time=time),
        ds[varname].sel(collection=set2).isel(time=time),
        ['lat', 'lon'],
    )

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
        print('WARNING: {} of 4 tests failed.'.format(num_fail))

    return num_fail


def subset_data(ds, subset, lat=None, lon=None, lev=0, start=None, end=None):
    """
    Get a subset of the given dataArray, returns a dataArray
    """
    ds_subset = ds

    if start is not None and end is not None:
        ds_subset = ds_subset.isel(time=slice(start, end + 1))

    if subset == 'winter':
        ds_subset = ds_subset.where(ds.time.dt.season == 'DJF', drop=True)
    elif subset == 'spring':
        ds_subset = ds_subset.where(ds.time.dt.season == 'MAM', drop=True)
    elif subset == 'summer':
        ds_subset = ds_subset.where(ds.time.dt.season == 'JJA', drop=True)
    elif subset == 'autumn':
        ds_subset = ds_subset.where(ds.time.dt.season == 'SON', drop=True)

    elif subset == 'first5':
        ds_subset = ds_subset.isel(time=slice(None, 5))

    if 'lev' in ds_subset.dims:
        ds_subset = ds_subset.isel(lev=lev)

    if lat is not None:
        ds_subset = ds_subset.sel(lat=lat, method='nearest')
        ds_subset = ds_subset.expand_dims('lat')

    if lon is not None:
        ds_subset = ds_subset.sel(lon=lon + 180, method='nearest')
        ds_subset = ds_subset.expand_dims('lon')

    return ds_subset
