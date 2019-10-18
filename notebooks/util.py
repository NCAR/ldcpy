import numpy as np
from scipy import stats as stats

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
