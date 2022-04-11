import collections
import csv
import os

import cf_xarray as cf
import dask
import numpy as np
import xarray as xr

from .calcs import Datasetcalcs, Diffcalcs


def collect_datasets(data_type, varnames, list_of_ds, labels, **kwargs):
    """
    Concatonate several different xarray datasets across a new
    "collection" dimension, which can be accessed with the specified
    labels.  Stores them in an xarray dataset which can be passed to
    the ldcpy plot functions (Call this OR open_datasets() before
    plotting.)


    Parameters
    ==========
    data_type: string
        Current data types: :cam-fv, pop
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

    if data_type == 'cam-fv':
        weights_name = 'gw'
        varnames.append(weights_name)
    elif data_type == 'pop':
        weights_name = 'TAREA'
        varnames.append(weights_name)

    # preprocess_vars is here for working on jupyter hub...
    def preprocess_vars(ds, varnames):
        return ds[varnames]

    # preprocess
    for i, myds in enumerate(list_of_ds):
        list_of_ds[i] = preprocess_vars(myds, varnames)

    full_ds = xr.concat(list_of_ds, 'collection', **kwargs)

    if data_type == 'pop':
        full_ds.coords['cell_area'] = xr.DataArray(full_ds.variables.mapping.get(weights_name))[0]
    else:
        full_ds.coords['cell_area'] = (
            xr.DataArray(full_ds.variables.mapping.get(weights_name))
            .expand_dims(lon=full_ds.dims['lon'])
            .transpose()
        )

    full_ds.attrs['cell_measures'] = 'area: cell_area'

    full_ds = full_ds.drop(weights_name)

    full_ds['collection'] = xr.DataArray(labels, dims='collection')

    print('dataset size in GB {:0.2f}\n'.format(full_ds.nbytes / 1e9))
    full_ds.attrs['data_type'] = data_type
    full_ds.attrs['file_size'] = None

    return full_ds


def combine_datasets(ds_list):
    new_ds = ds_list[0]
    for ds in ds_list[1:]:
        for var in ds.data_vars.variables.mapping:
            new_ds[var] = ds[var]
    return new_ds


def open_datasets(data_type, varnames, list_of_files, labels, weights=True, **kwargs):
    """
    Open several different netCDF files, concatenate across
    a new 'collection' dimension, which can be accessed with the specified
    labels. Stores them in an xarray dataset which can be passed to the ldcpy
    plot functions.

    Parameters
    ==========
    data_type: string
        Current data types: :cam-fv, pop
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
    file_size_dict = {}
    for i, myfile in enumerate(list_of_files):
        myds = xr.open_dataset(myfile)
        sz[i] = myds.sizes['time']
        myds.close()
        fs = os.path.getsize(myfile)
        file_size_dict[labels[i]] = fs
    indx = np.unique(sz)
    assert indx.size == 1, 'ERROR: all files must have the same length time dimension'

    # preprocess_vars is here for working on jupyter hub...
    def preprocess_vars(ds):
        return ds[varnames]

    if data_type == 'cam-fv' and weights is True:
        weights_name = 'gw'
        varnames.append(weights_name)
    elif data_type == 'pop' and weights is True:
        weights_name = 'TAREA'
        varnames.append(weights_name)

    full_ds = xr.open_mfdataset(
        list_of_files,
        concat_dim='collection',
        combine='nested',
        data_vars=varnames,
        parallel=True,
        preprocess=preprocess_vars,
        **kwargs,
    )

    if data_type == 'pop' and weights is True:
        full_ds.coords['cell_area'] = xr.DataArray(full_ds.variables.mapping.get(weights_name))[0]
    elif weights is True:
        full_ds.coords['cell_area'] = (
            xr.DataArray(full_ds.variables.mapping.get(weights_name))
            .expand_dims(lon=full_ds.dims['lon'])
            .transpose()
        )

    full_ds.attrs['cell_measures'] = 'area: cell_area'

    if weights is True:
        full_ds = full_ds.drop(weights_name)

    full_ds['collection'] = xr.DataArray(labels, dims='collection')
    print('dataset size in GB {:0.2f}\n'.format(full_ds.nbytes / 1e9))
    full_ds.attrs['data_type'] = data_type
    full_ds.attrs['file_size'] = file_size_dict

    for v in varnames[:-1]:
        new_ds = []
        i = 0
        for label in labels:
            new_ds.append(full_ds[v].sel(collection=label))
            new_ds[i].attrs['data_type'] = data_type
            new_ds[i].attrs['set_name'] = label

        # d = xr.combine_by_coords(new_ds)
        d = xr.concat(new_ds, 'collection')
        full_ds[v] = d

    return full_ds


def compare_stats(
    ds,
    varname: str,
    sets,
    significant_digits: int = 5,
    include_ssim: bool = False,
    weighted: bool = True,
    **calcs_kwargs,
):
    """
    Print error summary statistics for multiple DataArrays (should just be a single time slice)

    Parameters
    ==========
    ds : xarray.Dataset
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname : str
        The variable of interest in the dataset
    sets: list of str
        The labels of the collection to compare (all will be compared to the first set)
    significant_digits : int, optional
        The number of significant digits to use when printing stats (default 5)
    include_ssim : bool, optional
        Whether or not to compute the image ssim - slow for 3D vars (default: False)
    weighted : bool, optional
        Whether or not weight the means (default = True)
    **calcs_kwargs :
        Additional keyword arguments passed through to the
        :py:class:`~ldcpy.Datasetcalcs` instance.

    Returns
    =======
    out : None

    """

    # get a datarray for the variable of interest and get collections
    # (this is done seperately to work with cf_xarray)

    da = ds[varname]
    data_type = ds.attrs['data_type']

    file_size_dict = ds.attrs['file_size']
    if file_size_dict is None:
        include_file_size = False
    else:
        include_file_size = True

    da.attrs['cell_measures'] = 'area: cell_area'

    # use this after the update instead of above
    # da = ds.cf.data_vars[varname]

    # do we have more than one time slice? SHould only have one..
    if 'time' in da.dims:
        print('Warning - this data set has a time dimension - examining slice 0 only...')
        da = da.isel(time=0)

    # see how many sets we have
    da_sets = []

    num = len(sets)
    if num < 2:
        print('Error: must specify at least two sets to compare!')
        return
    for set in sets:
        da_sets.append(da.sel(collection=set))

    dd_sets = []
    for i in range(1, num):
        dd_sets.append(da_sets[0] - da_sets[i])

    aggregate_dims = calcs_kwargs.pop('aggregate_dims', None)

    da_set_calcs = []
    for i in range(num):
        da_set_calcs.append(
            Datasetcalcs(da_sets[i], data_type, aggregate_dims, **calcs_kwargs, weighted=weighted)
        )

    dd_set_calcs = []
    for i in range(num - 1):
        dd_set_calcs.append(
            Datasetcalcs(dd_sets[i], data_type, aggregate_dims, **calcs_kwargs, weighted=weighted)
        )

    diff_calcs = []
    for i in range(1, num):
        diff_calcs.append(
            Diffcalcs(
                da_sets[0], da_sets[i], data_type, aggregate_dims, **calcs_kwargs, weighted=weighted
            )
        )

    # DATA FRAME
    import pandas as pd
    from IPython.display import HTML, display

    df_dict = {}
    my_cols = []
    for i in range(num):
        my_cols.append(sets[i])

    temp_mean = []
    temp_var = []
    temp_std = []
    temp_min = []
    temp_max = []
    temp_pos = []
    temp_zeros = []
    temp_ac_lat = []
    temp_ac_lon = []
    temp_entropy = []

    for i in range(num):
        temp_mean.append(da_set_calcs[i].get_calc('mean').data.compute())
        temp_var.append(da_set_calcs[i].get_calc('variance').data.compute())
        temp_std.append(da_set_calcs[i].get_calc('std').data.compute())
        temp_max.append(da_set_calcs[i].get_calc('max_val').data.compute())
        temp_min.append(da_set_calcs[i].get_calc('min_val').data.compute())
        temp_pos.append(da_set_calcs[i].get_calc('prob_positive').data.compute())
        temp_zeros.append(da_set_calcs[i].get_calc('num_zero').data.compute())
        if data_type == 'cam-fv':
            temp_ac_lat.append(da_set_calcs[i].get_single_calc('lat_autocorr'))
            temp_ac_lon.append(da_set_calcs[i].get_single_calc('lon_autocorr'))
        temp_entropy.append(da_set_calcs[i].get_single_calc('entropy'))

    df_dict['mean'] = temp_mean
    df_dict['variance'] = temp_var
    df_dict['standard deviation'] = temp_std
    df_dict['min value'] = temp_min
    df_dict['max value'] = temp_max
    df_dict['probability positive'] = temp_pos
    df_dict['number of zeros'] = temp_zeros
    if data_type == 'cam-fv':
        df_dict['spatial autocorr - latitude'] = temp_ac_lat
        df_dict['spatial autocorr - longitude'] = temp_ac_lon
    df_dict['entropy estimate'] = temp_entropy

    for d in df_dict.keys():
        fo = [f'%.{significant_digits}g' % item for item in df_dict[d]]
        df_dict[d] = fo
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=my_cols)
    display(HTML(' <span style="color:green">Comparison: </span>  '))
    display(df)

    # diff stuff
    df_dict2 = {}
    my_cols2 = []
    for i in range(1, num):
        my_cols2.append(sets[i])

    temp_max_abs = []
    temp_min_abs = []
    temp_mean_abs = []
    temp_mean_sq = []
    temp_rms = []

    for i in range(num - 1):
        temp_max_abs.append(dd_set_calcs[i].get_calc('max_abs').data.compute())
        temp_min_abs.append(dd_set_calcs[i].get_calc('min_abs').data.compute())
        temp_mean_abs.append(dd_set_calcs[i].get_calc('mean_abs').data.compute())
        temp_mean_sq.append(dd_set_calcs[i].get_calc('mean_squared').data.compute())
        temp_rms.append(dd_set_calcs[i].get_calc('rms').data.compute())

    df_dict2['max abs diff'] = temp_max_abs
    df_dict2['min abs diff'] = temp_min_abs
    df_dict2['mean abs diff'] = temp_mean_abs
    df_dict2['mean squared diff'] = temp_mean_sq
    df_dict2['root mean squared diff'] = temp_rms

    temp_nrms = []
    temp_max_pe = []
    temp_pcc = []
    temp_ks = []
    temp_sre = []
    temp_max_spr = []
    temp_data_ssim = []
    temp_ssim = []
    temp_cr = []

    # compare to the first set
    if include_file_size:
        fs_orig = file_size_dict[sets[0]]

    for i in range(num - 1):
        temp_nrms.append(diff_calcs[i].get_diff_calc('n_rms').data.compute())
        temp_max_pe.append(diff_calcs[i].get_diff_calc('n_emax').data.compute())
        temp_pcc.append(
            diff_calcs[i].get_diff_calc('pearson_correlation_coefficient').data.compute()
        )
        temp_ks.append(diff_calcs[i].get_diff_calc('ks_p_value'))
        temp_sre.append(diff_calcs[i].get_diff_calc('spatial_rel_error'))
        temp_max_spr.append(diff_calcs[i].get_diff_calc('max_spatial_rel_error'))
        temp_data_ssim.append(diff_calcs[i].get_diff_calc('ssim_fp'))
        if include_ssim:
            temp_ssim.append(diff_calcs[i].get_diff_calc('ssim'))

        if include_file_size:
            this_fs = file_size_dict[my_cols2[i]]
            temp_cr.append(round(fs_orig / this_fs, 2))

    df_dict2['normalized root mean squared diff'] = temp_nrms
    df_dict2['normalized max pointwise error'] = temp_max_pe
    df_dict2['pearson correlation coefficient'] = temp_pcc
    df_dict2['ks p-value'] = temp_ks

    tmp_str = 'spatial relative error(% > ' + str(da_set_calcs[0].get_single_calc('spre_tol')) + ')'
    df_dict2[tmp_str] = temp_sre

    df_dict2['max spatial relative error'] = temp_max_spr
    df_dict2['data SSIM'] = temp_data_ssim
    if include_ssim:
        df_dict2['image SSIM'] = temp_ssim

    if include_file_size:
        df_dict2['file size ratio'] = temp_cr

    for d in df_dict2.keys():
        fo = [f'%.{significant_digits}g' % item for item in df_dict2[d]]
        df_dict2[d] = fo

    df2 = pd.DataFrame.from_dict(df_dict2, orient='index', columns=my_cols2)

    display(HTML('<br>'))
    display(HTML('<span style="color:green">Difference calcs: </span>  '))
    display(df2)


def check_metrics(
    ds,
    varname,
    set1,
    set2,
    ks_tol=0.05,
    pcc_tol=0.99999,
    spre_tol=5.0,
    ssim_tol=0.995,
    **calcs_kwargs,
):
    """

    Check the K-S, Pearson Correlation, and Spatial Relative Error calcs

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
         The threshold for the data ssim test (default = .995
    **calcs_kwargs :
        Additional keyword arguments passed through to the
        :py:class:`~ldcpy.Datasetcalcs` instance.

    Returns
    =======
    out : Number of failing calcs

    Notes
    ======

    Check the K-S, Pearson Correlation, and Spatial Relative Error calcs from:

    A. H. Baker, H. Xu, D. M. Hammerling, S. Li, and J. Clyne,
    “Toward a Multi-method Approach: Lossy Data Compression for
    Climate Simulation Data”, in J.M. Kunkel et al. (Eds.): ISC
    High Performance Workshops 2017, Lecture Notes in Computer
    Science 10524, pp. 30–42, 2017 (doi:10.1007/978-3-319-67630-2_3).

    Check the Data SSIM, which is a modification of SSIM calc from:

    A.H. Baker, D.M. Hammerling, and T.L. Turton. “Evaluating image
    quality measures to assess the impact of lossy data compression
    applied to climate simulation data”, Computer Graphics Forum 38(3),
    June 2019, pp. 517-528 (doi:10.1111/cgf.13707).

    Default tolerances for the tests are:
    ------------------------
    K-S: fail if p-value < .05 (significance level)
    Pearson correlation coefficient:  fail if coefficient < .99999
    Spatial relative error: fail if > 5% of grid points fail relative error
    Data SSIM: fail if Data SSIM < .995

    """

    print(f'Evaluating 4 calcs for {set1} data (set1) and {set2} data (set2):')
    data_type = ds.attrs['data_type']
    aggregate_dims = calcs_kwargs.pop('aggregate_dims', None)
    diff_calcs = Diffcalcs(
        ds[varname].sel(collection=set1),
        ds[varname].sel(collection=set2),
        data_type,
        aggregate_dims,
        **calcs_kwargs,
        weighted=False,
    )

    # count the number of failures
    num_fail = 0
    # Pearson less than pcc_tol means fail
    pcc = diff_calcs.get_diff_calc('pearson_correlation_coefficient').data.compute()
    if pcc < pcc_tol:
        print('     *FAILED pearson correlation coefficient test...(pcc = {0:.5f}'.format(pcc), ')')
        num_fail = num_fail + 1
    else:
        print('     PASSED pearson correlation coefficient test...(pcc = {0:.5f}'.format(pcc), ')')
    # K-S p-value less than ks_tol means fail (can reject null hypo)
    ks = diff_calcs.get_diff_calc('ks_p_value')
    if ks < ks_tol:
        print('     *FAILED ks test...(ks p_val = {0:.4f}'.format(ks), ')')
        num_fail = num_fail + 1
    else:
        print('     PASSED ks test...(ks p_val = {0:.4f}'.format(ks), ')')
    # Spatial rel error fails if more than spre_tol
    spre = diff_calcs.get_diff_calc('spatial_rel_error')
    if spre > spre_tol:
        print('     *FAILED spatial relative error test ... (spre = {0:.2f}'.format(spre), ' %)')
        num_fail = num_fail + 1
    else:
        print('     PASSED spatial relative error test ...(spre = {0:.2f}'.format(spre), ' %)')
    # SSIM less than of ssim_tol is failing
    ssim_val = diff_calcs.get_diff_calc('ssim_fp')
    if ssim_val < ssim_tol:
        print('     *FAILED DATA SSIM test ... (ssim = {0:.5f}'.format(ssim_val), ')')
        num_fail = num_fail + 1
    else:
        print('     PASSED DATA SSIM test ... (ssim = {0:.5f}'.format(ssim_val), ')')
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
        if subset == 'DJF':
            ds_subset = ds_subset.cf.sel(time=ds.cf['time'].dt.season == 'DJF')
        elif subset == 'MAM':
            ds_subset = ds_subset.cf.sel(time=ds.cf['time'].dt.season == 'MAM')
        elif subset == 'JJA':
            ds_subset = ds_subset.cf.sel(time=ds.cf['time'].dt.season == 'JJA')
        elif subset == 'SON':
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


def var_and_wt_coords(varname, ds_col):

    ca_coord = ds_col.coords['cell_area']
    if dask.is_dask_collection(ca_coord):
        ca_coord = ca_coord.compute()
    ds0 = ds_col.cf[varname]
    all_coords = ds0.coords
    all_coords['cell_area'] = ca_coord
    ds0.assign_coords(all_coords)
    ds0.attrs['cell_measures'] = 'area: cell_area'

    return ds0


def save_metrics(
    full_ds,
    varname,
    set1,
    set2,
    time=0,
    lev=0,
    location='names.csv',
):
    """
    full_ds : xarray.Dataset
        An xarray dataset containing multiple netCDF files concatenated across a 'collection' dimension
    varname : str
        The variable of interest in the dataset
    set1 : str
        The collection label of the "control" data
    set2 : str
        The collection label of the (1st) data to compare
    time : int, optional
        The time index used t (default = 0)
    time : lev, optional
        The level index of interest in a 3D dataset (default 0)
    Returns
    =======
    out : Number of failing metrics
    """

    ds = subset_data(full_ds)

    # count the number of failuress
    num_fail = 0

    print(
        'Evaluating metrics for {} data (set1) and {} data (set2), time {}'.format(
            set1, set2, time
        ),
        ':',
    )

    diff_metrics = Diffcalcs(
        ds[varname].sel(collection=set1).isel(time=time),
        ds[varname].sel(collection=set2).isel(time=time),
        'cam-fv',
        ['lat', 'lon'],
    )

    # SSIM
    ssim_fp_val = diff_metrics.get_diff_calc('ssim_fp')

    file_exists = os.path.isfile(location)
    with open(location, 'a', newline='') as csvfile:
        fieldnames = [
            'set',
            'time',
            'ssim_fp',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                'set': set2,
                'time': time,
                'ssim_fp': ssim_fp_val,
            }
        )

    return num_fail
