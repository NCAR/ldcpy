import xarray as xr

from .error_metrics import ErrorMetrics


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

    Keywork Arguments:
    ==================
    time -- int
        the time index used to compare the two netCDF files (default 0)

    Returns
    =======
    out -- None

    """
    print('Comparing {} data to {} data'.format(ens_o, ens_r))
    orig_val = ds[varname].sel(ensemble=ens_o).isel(time=time)
    recon_val = ds[varname].sel(ensemble=ens_r).isel(time=time)

    em = ErrorMetrics(orig_val.values, recon_val.values)

    import json

    print(
        json.dumps(
            em.get_all_metrics({'error', 'squared_error', 'absolute_error'}),
            indent=4,
            separators=(',', ': '),
        )
    )
