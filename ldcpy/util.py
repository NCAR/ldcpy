import xarray as xr

from .metrics import DatasetMetrics, DiffMetrics


def open_datasets(list_of_files, ensemble_names, pot_var_names=['TS', 'PRECT', 'T']):
    """
    Open several different netCDF files, concatenate across
    a new 'ensemble' dimension. Stores them in an xarray dataset.

    Parameters:
    ===========
    list_of_files -- list <string>
        the path of the netCDF file(s) to be opened
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


def print_stats(ds, varname, ens_o, ens_r, time=0):
    """
    Print error summary statistics of two DataArrays

    Parameters:
    ===========
    ds -- xarray.Dataset
        an xarray dataset containing multiple netCDF files concatenated across an 'ensemble' dimension
    varname -- string
        the variable of interest in the dataset
    ens_o -- string
        the ensemble label of the original data
    ens_r -- string
        the ensemble label of the reconstructed data

    Keyword Arguments:
    ==================
    time -- int
        the time index used to compare the two netCDF files (default 0)

    Returns
    =======
    out -- None

    """
    print('Comparing {} data to {} data'.format(ens_o, ens_r))

    import json

    ds1_metrics = DatasetMetrics(ds[varname].sel(ensemble=ens_o).isel(time=time), ['lat', 'lon'])
    ds2_metrics = DatasetMetrics(ds[varname].sel(ensemble=ens_r).isel(time=time), ['lat', 'lon'])
    d_metrics = DatasetMetrics(
        ds[varname].sel(ensemble=ens_o).isel(time=time)
        - ds[varname].sel(ensemble=ens_r).isel(time=time),
        ['lat', 'lon'],
    )
    diff_metrics = DiffMetrics(
        ds[varname].sel(ensemble=ens_o).isel(time=time),
        ds[varname].sel(ensemble=ens_r).isel(time=time),
        ['lat', 'lon'],
    )

    output = {}
    output['mean_observed'] = ds1_metrics.get_metric('mean').item(0)
    output['variance_observed'] = ds1_metrics.get_metric('variance').item(0)
    output['standard deviation observed'] = ds1_metrics.get_metric('std').item(0)

    output['mean modelled'] = ds2_metrics.get_metric('mean').item(0)
    output['variance modelled'] = ds2_metrics.get_metric('variance').item(0)
    output['standard deviation modelled'] = ds2_metrics.get_metric('std').item(0)

    d_metrics.quantile = 1
    output['max diff'] = d_metrics.get_metric('quantile').item(0)
    d_metrics.quantile = 0
    output['min diff'] = d_metrics.get_metric('quantile').item(0)
    output['mean squared diff'] = d_metrics.get_metric('mean_squared').item(0)
    output['mean diff'] = d_metrics.get_metric('mean').item(0)
    output['mean abs diff'] = d_metrics.get_metric('mean_abs').item(0)
    output['root mean squared diff'] = d_metrics.get_metric('rms').item(0)

    output['pearson correlation coefficient'] = diff_metrics.get_diff_metric(
        'pearson_correlation_coefficient'
    ).item(0)
    output['covariance'] = diff_metrics.get_diff_metric('covariance').item(0)
    output['ks p value'] = diff_metrics.get_diff_metric('ks_p_value').item(0)

    print(json.dumps(output, indent=4, separators=(',', ': '),))


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
