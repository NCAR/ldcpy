import xarray as xr
import numpy as np
from scipy import stats as stats

###############

def open_datasets(list_of_files, ensemble_names):
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

    full_ds = xr.concat(ds_list, 'ensemble', data_vars=['TS'])
    full_ds['ensemble'] = xr.DataArray(ensemble_names, dims='ensemble')
    del ds_list

    return(full_ds)

###############

def print_stats(ds, varname, ens_o, ens_r, time=0):
    print('Comparing {} data to {} data'.format(ens_o, ens_r))
    orig_val = ds[varname].sel(ensemble=ens_o).isel(time=time)
    recon_val = ds[varname].sel(ensemble=ens_r).isel(time=time)
    print('KS = {}'.format(calc_ks(orig_val, recon_val)))
    print('corr = {}'.format(calc_corr(orig_val, recon_val)))
    print('nrmse = {}'.format(calc_nrmse(orig_val, recon_val)))
    print('mae = {}'.format(calc_mae(orig_val, recon_val)))
    print('me = {}'.format(calc_me(orig_val, recon_val)))
    print('me = {}'.format(calc_maxerr(orig_val, recon_val)))

###############

def calc_ks(orig_val, recon_val):
    """
    calculate K-S p-value
    """
    o64 = orig_val.astype(np.float64)
    r64 = recon_val.astype(np.float64)
    ks_stat, p_val = stats.ks_2samp(np.ravel(o64), np.ravel(r64))
    return p_val

###############

def calc_corr(orig_val, recon_val):
    """
    pearson correlation coefficient
    """
    o64 = orig_val.astype(np.float64)
    r64 = recon_val.astype(np.float64)
    r, pval = stats.pearsonr(np.ravel(o64), np.ravel(r64))
    return r

###############

def calc_nrmse(orig_val, recon_val):
    """
    normalized (by range) root mean square error
    """
    e = _error_d(orig_val, recon_val)
    mse = np.sum(np.square(e)).mean()
    rmse = np.sqrt(mse)
    dy_range = np.max(orig_val)-np.min(orig_val)
    nrmse = rmse/dy_range
    #if you want to return a float
    return nrmse.data.item()

###############

def calc_mae(orig_val, recon_val):
    """
    mean absolute error (not weighted - yet)
    """
    e = _error_d(orig_val, recon_val)
    return (np.mean(np.abs(e))).data.item()

###############

def calc_me(orig_val, recon_val):
    """
    mean error
    """
    e = _error_d(orig_val, recon_val)
    return (np.mean(e)).data.item()

###############

def calc_maxerr(orig_val, recon_val):
    """
    max error
    """
    e = _error_d(orig_val, recon_val)
    return (np.max(e)).data.item()

###############

def _error_d(orig_val, recon_val):
    """
    error in double precision
    """
    return orig_val.astype(np.float64) - recon_val.astype(np.float64)
