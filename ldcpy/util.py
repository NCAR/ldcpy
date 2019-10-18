import xarray as xr
from .error_metrics import ErrorMetrics


def open_datasets(list_of_files, ensemble_names, pot_var_names=['TS', 'PRECT']):
    """
    Open several different netCDF files, concanate across
    a new 'ensemble' dimension
    """

    # Error checking:
    # list_of_files and ensemble_names must be same length
    assert len(list_of_files) == len(ensemble_names), "open_dataset arguments must be same length"

    ds_list = []
    for filename in list_of_files:
        ds_list.append(xr.open_dataset(filename))

    data_vars = []
    for varname in pot_var_names:
        if varname in ds_list[0]:
            data_vars.append(varname)
    assert data_vars != [], "can not find any of {} in dataset".format(pot_var_names)
    full_ds = xr.concat(ds_list, 'ensemble', data_vars=data_vars)
    full_ds['ensemble'] = xr.DataArray(ensemble_names, dims='ensemble')
    del ds_list

    return(full_ds)


def print_stats(ds, varname, ens_o, ens_r, time=0):
    print('Comparing {} data to {} data'.format(ens_o, ens_r))
    orig_val = ds[varname].sel(ensemble=ens_o).isel(time=time)
    recon_val = ds[varname].sel(ensemble=ens_r).isel(time=time)

    em = ErrorMetrics(orig_val.values, recon_val.values)

    import json
    print(json.dumps(em.get_all_metrics({"error", "squared_error", "absolute_error"}), indent=4, separators=(",", ": ")))


