import xarray as xr

from .metrics import DatasetMetrics, DiffMetrics


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
        the file paths for the netCDF file(s) to be opened

    labels -- list <string>
        the respective label to access data from each netCDF file (also used in plotting fcns)

    **kwargs (optional) – Additional arguments passed on to xarray.open_mfdataset(). A list of available arguments can
    be found here: http://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html

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
        print("chucks set to (default) {'time', 50}")
        kwargs['chunks'] = {'time': 50}
    else:
        print('chunks set to (by user) ', kwargs['chunks'])

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


def print_stats(ds, varname, set1, set2, time=0, sig_dig=4):
    """
    Print error summary statistics of two DataArrays

    Parameters:
    ===========
    ds -- xarray.Dataset
        an xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname -- string
        the variable of interest in the dataset
    set1 -- string
        the collection label of the "control" data
    set2 -- string
        the collection label of the (1st) data to compare

    Keyword Arguments:
    ==================
    time -- int
        the time index used to compare the two netCDF files (default 0)

    Returns
    =======
    out -- None

    """
    print('Comparing {} data (set1) to {} data (set2)'.format(set1, set2))

    import json

    ds0_metrics = DatasetMetrics(ds[varname].sel(collection=set1).isel(time=time), ['lat', 'lon'])
    ds1_metrics = DatasetMetrics(ds[varname].sel(collection=set2).isel(time=time), ['lat', 'lon'])
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

    output = {}

    output['skip1'] = 0

    output['mean set1'] = ds0_metrics.get_metric('mean').values
    output['mean set2'] = ds1_metrics.get_metric('mean').values
    output['mean diff'] = d_metrics.get_metric('mean').values

    output['skip2'] = 0

    output['variance set1'] = ds0_metrics.get_metric('variance').values
    output['variance set2'] = ds1_metrics.get_metric('variance').values

    output['skip3'] = 0

    output['standard deviation set1'] = ds0_metrics.get_metric('std').values
    output['standard deviation set2'] = ds1_metrics.get_metric('std').values

    output['skip4'] = 0

    # output['dynamic range set1'] = ds0_metrics.get_metric('range').values
    # output['dynamic range set2'] = ds1_metrics.get_metric('range').values

    # output['skip5'] = 0

    output['max value set1'] = ds0_metrics.get_metric('max_val').values
    output['max value set2'] = ds1_metrics.get_metric('max_val').values
    output['min value set1'] = ds0_metrics.get_metric('min_val').values
    output['min value set2'] = ds1_metrics.get_metric('min_val').values

    output['skip55'] = 0

    output['max abs diff'] = d_metrics.get_metric('max_abs').values
    output['min abs diff'] = d_metrics.get_metric('min_abs').values
    output['mean abs diff'] = d_metrics.get_metric('mean_abs').values

    output['mean squared diff'] = d_metrics.get_metric('mean_squared').values
    output['root mean squared diff'] = d_metrics.get_metric('rms').values

    output['normalized root mean squared diff'] = diff_metrics.get_diff_metric('n_rms').values
    output['normalized max pointwise error'] = diff_metrics.get_diff_metric('n_emax').values
    output['pearson correlation coefficient'] = diff_metrics.get_diff_metric(
        'pearson_correlation_coefficient'
    ).values
    output['ks p-value'] = diff_metrics.get_diff_metric('ks_p_value')[0]

    for key, value in output.items():
        if key[:4] != 'skip':
            print(f'{key}: {value:.{sig_dig}e}')
        else:
            print(' ')


#    print(json.dumps(output, indent=4, separators=(',', ': '),))


def subset_data(ds, subset, lat=None, lon=None, lev=0, start=None, end=None):
    """
    Get a subset of the given dataArray, returns a dataArray
    """
    ds_subset = ds

    ds_subset = ds_subset.isel(time=slice(start, end))

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
