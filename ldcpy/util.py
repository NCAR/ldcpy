import xarray as xr

from .metrics import DatasetMetrics, DiffMetrics


def orig_open_datasets(list_of_files, ensemble_names, pot_var_names=['TS', 'PRECT', 'T']):
    """
    Open several different netCDF files, concatenate across
    a new 'ensemble' dimension. Stores them in an xarray dataset.
    (Assuming timeseries files)

    Parameters:
    ===========
    list_of_files -- list <string>
        the path of the net CDF file(s) to be opened
    ensemble_names -- list <string>
        the respective ensemble names of each netCDF file

    Keyword Arguments:
    ==================
    pot_var_names -- list <string>
        the variables to load data from in each netCDF file

    Returns
    =======
    out -- xarray.Dataset
        contains data variables matching each pot_var_name found in the netCDF file
    """

    # Error checking:
    # list_of_files and ensemble_names must be same length
    assert len(list_of_files) == len(ensemble_names), 'open_dataset arguments must be same length'

    ds_list = []
    for filename in list_of_files:
        ds_list.append(xr.open_dataset(filename))

    data_vars = []
    for varname in pot_var_names:
        if varname in ds_list[0]:
            data_vars.append(varname)
    assert data_vars != [], 'can not find any of {} in dataset'.format(pot_var_names)
    full_ds = xr.concat(ds_list, 'ensemble', data_vars=data_vars)
    full_ds['ensemble'] = xr.DataArray(ensemble_names, dims='ensemble')
    del ds_list

    return full_ds


def open_datasets(varnames, list_of_files, labels, **kwargs):
    """
    Open several different netCDF files, concatenate across
    a new 'collection' dimension, which can be accessed with labels.
    Stores them in an xarray dataset.

    Parameters:
    ===========
    varnames -- list <string>
           the variable(s) of interest to combine across input files (usually just one)

    list_of_files -- list <string>
        the path of the netCDF file(s) to be opened

    labels -- list <string>
        the respective label to access data from each netCDF file (also used in plotting fcns)

    **kwargs (optional) – Additional arguments passed on to xarray.open_mfdataset().

    Returns
    =======
    out -- xarray.Dataset
          contains all the data from the list of files
    """

    # Error checking:
    # list_of_files and ensemble_names must be same length
    assert len(list_of_files) == len(
        labels
    ), 'open_dataset file list and labels arguments must be the same length'

    # check whether we need to set chunks or the user has already done so
    if 'chunks' not in kwargs:
        print("chucks set to {'time', 50}")
        kwargs['chunks'] = {'time': 50}

    # check that varname exists in each file
    for filename in list_of_files:
        ds_check = xr.open_dataset(filename)
        for thisvar in varnames:
            if thisvar not in ds_check.variables:
                print(f"We have a problem. Variable '{thisvar}' is not in the file {filename}")
        ds_check.close()

    full_ds = xr.open_mfdataset(
        list_of_files, concat_dim='collection', combine='nested', data_vars=varnames, **kwargs,
    )

    full_ds['collection'] = xr.DataArray(labels, dims='collection')

    print('dataset size in GB {:0.2f}\n'.format(full_ds.nbytes / 1e9))

    return full_ds


def print_stats(ds, varname, c0, c1, time=0):
    """
    Print error summary statistics of two DataArrays

    Parameters:
    ===========
    ds -- xarray.Dataset
        an xarray dataset containing multiple netCDF files concatenated across an 'ensemble' dimension
    varname -- string
        the variable of interest in the dataset
    c0 -- string
        the collection label of the "control" data
    c1 -- string
        the collection label of the (1st) data to compare

    Keyword Arguments:
    ==================
    time -- int
        the time index used to compare the two netCDF files (default 0)

    Returns
    =======
    out -- None

    """
    print('Comparing {} data (c0) to {} data (c1)'.format(c0, c1))

    import json

    ds0_metrics = DatasetMetrics(ds[varname].sel(collection=c0).isel(time=time), ['lat', 'lon'])
    ds1_metrics = DatasetMetrics(ds[varname].sel(collection=c1).isel(time=time), ['lat', 'lon'])
    d_metrics = DatasetMetrics(
        ds[varname].sel(collection=c0).isel(time=time)
        - ds[varname].sel(collection=c1).isel(time=time),
        ['lat', 'lon'],
    )
    diff_metrics = DiffMetrics(
        ds[varname].sel(collection=c0).isel(time=time),
        ds[varname].sel(collection=c1).isel(time=time),
        ['lat', 'lon'],
    )

    output = {}
    output['mean_control'] = ds0_metrics.get_metric('mean').values
    output['variance_control'] = ds0_metrics.get_metric('variance').values
    output['standard deviation control'] = ds0_metrics.get_metric('std').values

    output['mean c1'] = ds1_metrics.get_metric('mean').values
    output['variance c1'] = ds1_metrics.get_metric('variance').values
    output['standard deviation c1'] = ds1_metrics.get_metric('std').values

    d_metrics.quantile = 1
    output['max diff'] = d_metrics.get_metric('quantile').values
    d_metrics.quantile = 0
    output['min diff'] = d_metrics.get_metric('quantile').values
    output['mean squared diff'] = d_metrics.get_metric('mean_squared').values
    output['mean diff'] = d_metrics.get_metric('mean').values
    output['mean abs diff'] = d_metrics.get_metric('mean_abs').values
    output['root mean squared diff'] = d_metrics.get_metric('rms').values

    output['pearson correlation coefficient'] = diff_metrics.get_diff_metric(
        'pearson_correlation_coefficient'
    ).values
    output['covariance'] = diff_metrics.get_diff_metric('covariance').values
    output['ks p value'] = diff_metrics.get_diff_metric('ks_p_value')[1]

    [print('          ', key, ': ', value) for key, value in output.items()]


#    print(json.dumps(output, indent=4, separators=(',', ': '),))


def subset_data(ds, subset, lat=None, lon=None, lev=0, start=None, end=None):
    """
    Get a
    """
    ds_subset = ds

    ds_subset = ds_subset.isel(time=slice(start, end))

    if subset == 'winter':
        ds_subset = ds_subset.where(ds.time.dt.season == 'DJF', drop=True)
    elif subset == 'first50':
        ds_subset = ds_subset.isel(time=slice(None, 50))

    if 'lev' in ds_subset.dims:
        ds_subset = ds_subset.sel(lev=lev, method='nearest')

    if lat is not None:
        ds_subset = ds_subset.sel(lat=lat, method='nearest')
        ds_subset = ds_subset.expand_dims('lat')

    if lon is not None:
        ds_subset = ds_subset.sel(lon=lon + 180, method='nearest')
        ds_subset = ds_subset.expand_dims('lon')

    return ds_subset
