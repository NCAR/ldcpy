import cf_xarray as cf
import dask
import matplotlib as mpl
import numpy as np
import xarray as xr

from ldcpy import util as lu

xr.set_options(keep_attrs=True)


def _preprocess(set_labels, list_of_cols):

    contU = True
    num_sets = len(set_labels)

    if num_sets < 2:
        print('Error: Must have at least 2 set labels')
        contU = False
        return contU

    # Add a check: need at least one year of data

    return contU


# top of the model radiation budget
def cam_restom(fsnt_col, flnt_col, sets):

    col = []

    fsnt = fsnt_col['FSNT']
    flnt = flnt_col['FLNT']

    fsnt.attrs['cell_measures'] = 'area: cell_area'
    flnt.attrs['cell_measures'] = 'area: cell_area'

    col.append(fsnt)
    col.append(flnt)

    contU = _preprocess(sets, col)
    if not contU:
        return

    num_sets = len(sets)

    fsnt_data = []
    flnt_data = []
    for s in sets:
        fsnt_data.append(fsnt.sel(collection=s))
        flnt_data.append(flnt.sel(collection=s))

    # Now the calculation
    out_array = np.zeros(num_sets)
    percent_diff = np.zeros(num_sets - 1)
    for j in range(num_sets):
        # need to normalize by area
        tmp_data = fsnt_data[j] - flnt_data[j]
        tmp = tmp_data.cf.weighted('area').mean()
        # output
        out_array[j] = tmp
        if j == 0:
            control = tmp
            if control == 0:
                control = 1
        else:
            percent_diff[j - 1] = np.abs((control - tmp) / control) * 100

    print('values = ', out_array)
    print('percent rel. difference = ', percent_diff)


# global precipitation
def cam_precip(precc_col, precl_col, sets):

    col = []

    precc = precc_col['PRECC']
    precl = precl_col['PRECL']

    precc.attrs['cell_measures'] = 'area: cell_area'
    precl.attrs['cell_measures'] = 'area: cell_area'

    col.append(precc)
    col.append(precl)

    contU = _preprocess(sets, col)
    if not contU:
        return

    num_sets = len(sets)

    precc_data = []
    precl_data = []
    for s in sets:
        precc_data.append(precc.sel(collection=s))
        precl_data.append(precl.sel(collection=s))

    # Now the calculation
    out_array = np.zeros(num_sets)
    percent_diff = np.zeros(num_sets - 1)
    for j in range(num_sets):
        tmp_data = precc_data[j] + precl_data[j]
        tmp = tmp_data.cf.weighted('area').mean()
        out_array[j] = tmp
        if j == 0:
            control = tmp
            if control == 0:
                control = 1
        else:
            percent_diff[j - 1] = np.abs((control - tmp) / control) * 100

    print('values = ', out_array)
    print('percent rel. difference = ', percent_diff)


# evaporation-precipitation
def cam_ep(qflx_col, precc_col, precl_col, sets):

    # QFLX is "kg/m2/s or mm/s
    # PRECC and PRECL are m/s
    # 1 kg/m2/s = 86400 mm/day.
    # 8.64e7 mm/day = 1 m/sec

    col = []

    qflx = qflx_col['QFLX']
    precc = precc_col['PRECC']
    precl = precl_col['PRECL']

    qflx.attrs['cell_measures'] = 'area: cell_area'
    precc.attrs['cell_measures'] = 'area: cell_area'
    precl.attrs['cell_measures'] = 'area: cell_area'

    col.append(qflx)
    col.append(precc)
    col.append(precl)

    contU = _preprocess(sets, col)
    if not contU:
        return

    num_sets = len(sets)

    qflx_data = []
    precc_data = []
    precl_data = []
    for s in sets:
        qflx_data.append(qflx.sel(collection=s))
        precc_data.append(precc.sel(collection=s))
        precl_data.append(precl.sel(collection=s))

    # Now the calculation
    out_array = np.zeros(num_sets)
    percent_diff = np.zeros(num_sets - 1)
    for j in range(num_sets):
        tmp_data = qflx_data[j] * 86400 + (precc_data[j] + precl_data[j]) * 8.64e7
        tmp = tmp_data.cf.weighted('area').mean()
        # output
        out_array[j] = tmp
        if j == 0:
            control = tmp
            if control == 0:
                control = 1
        else:
            percent_diff[j - 1] = np.abs((control - tmp) / control) * 100

    print('values = ', out_array)
    print('percent rel. difference = ', percent_diff)


# surface energy balance
def cam_ressurf(fsns_col, flns_col, shflx_col, lhflx_col, sets):

    col = []

    # all in W/m^2
    fsns = fsns_col['FSNS']
    flns = flns_col['FLNS']
    shflx = shflx_col['SHFLX']
    lhflx = lhflx_col['LHFLX']

    fsns.attrs['cell_measures'] = 'area: cell_area'
    flns.attrs['cell_measures'] = 'area: cell_area'
    shflx.attrs['cell_measures'] = 'area: cell_area'
    lhflx.attrs['cell_measures'] = 'area: cell_area'

    col.append(fsns)
    col.append(flns)
    col.append(shflx)
    col.append(lhflx)

    contU = _preprocess(sets, col)
    if not contU:
        return

    num_sets = len(sets)
    fsns_data = []
    flns_data = []
    shflx_data = []
    lhflx_data = []
    for s in sets:
        fsns_data.append(fsns.sel(collection=s))
        flns_data.append(flns.sel(collection=s))
        shflx_data.append(shflx.sel(collection=s))
        lhflx_data.append(lhflx.sel(collection=s))

    # Now the calculation
    out_array = np.zeros(num_sets)
    percent_diff = np.zeros(num_sets - 1)
    for j in range(num_sets):
        tmp_data = fsns_data[j] - (flns_data[j] + shflx_data[j] + lhflx_data[j])
        tmp = tmp_data.cf.weighted('area').mean()

        out_array[j] = tmp
        if j == 0:
            control = tmp
            if control == 0:
                control = 1
        else:
            percent_diff[j - 1] = np.abs((control - tmp) / control) * 100

    print('values = ', out_array)
    print('percent rel. difference = ', percent_diff)


def cam_budgets(
    all_data,
    sets,
    significant_digits: int = 5,
):
    """
    Print CAM budgets (required variables listed in parenthesis):

    (1) restom: top of the model radiation budget (FSNT, FLNT)
    (2) precip: global precipitation (PRECC, PRECL)
    (3) ep: evaporation-precipitation (QFLX, PRECC, PRECL)
    (4) ressurf: surface energy balance (FSNS, FLNS, SHFLX, LHFLX)


    Parameters
    ==========
    all_data : xarray.Dataset
        An xarray dataset containing multiple variables needed for budgets
    sets: list of str
        The labels of the collection to compare (all will be compared to the first set)
    significant_digits : int, optional
        The number of significant digits to use when printing budgets (default 5)
    Returns
    =======
    out : None

    """

    # get list of variable names and check for needed vars

    # DATA FRAME
    import pandas as pd
    from IPython.display import HTML, display

    num = len(sets)
    df_dict = {}
    my_cols = []
    for i in range(num):
        my_cols.append(sets[i])

    # restom (FSNT, FLNT)
    # restom_data = cam_restom(fsnt_col, flnt_col, sets)
    # df_dict['restom'] = restom_data

    # precip (PRECC, PRECL)
    # precip_data = cam_precip(precc_col, precl_col, sets)
    # df_dict['precip'] = precip_data

    # ep (QFLX, PRECC, PRECL)
    # ep_data = cam_ep(qflx_col, precc_col, precl_col, sets)
    # df_dict['e-p'] = ep_data

    # ressurf (FSNS, FLNS, SHFLX, LHFLX)
    # ressurf_data = cam_ressurf(fsns_col, flns_col, shflx_col, lhflx_col, sets)
    # df_dict['ressurf'] = ressurf_data

    for d in df_dict.keys():
        fo = [f'%.{significant_digits}g' % item for item in df_dict[d]]
        df_dict[d] = fo
    df = pd.DataFrame.from_dict(df_dict, orient='index', columns=my_cols)
    display(HTML(' <span style="color:green">CAM Budgets: </span>  '))
    display(df)
