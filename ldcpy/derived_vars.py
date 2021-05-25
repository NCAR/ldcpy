import cf_xarray as cf
import dask
import matplotlib as mpl
import numpy as np
import xarray as xr

from ldcpy import util as lu

xr.set_options(keep_attrs=True)


def _preprocess(list_set_labels, list_of_cols):
    contU = True

    num_sets = len(list_set_labels[0])

    if num_sets < 2:
        print('Error: Must have at least 2 set lables for each variable')
        contU = False
        return contU

    for labels in list_set_labels:
        num = len(labels)
        if num != num_sets:
            print('Error: all collections must have the same number of set labels.')
            contU = False
            return contU

    # Add a check: need at least one year of data

    return contU


# top of the model radiation budget
def cam_restom(fsnt_col, flnt_col, list_of_sets):

    col = []

    fsnt = fsnt_col['FSNT']
    flnt = flnt_col['FLNT']

    fsnt.attrs['cell_measures'] = 'area: cell_area'
    flnt.attrs['cell_measures'] = 'area: cell_area'

    col.append(fsnt)
    col.append(flnt)

    contU = _preprocess(list_of_sets, col)
    if not contU:
        return

    num_sets = len(list_of_sets[0])

    fsnt_data = []
    flnt_data = []
    for set in list_of_sets[0]:
        fsnt_data.append(fsnt.sel(collection=set))
    for set in list_of_sets[1]:
        flnt_data.append(flnt.sel(collection=set))

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
def cam_precip(precc_col, precl_col, list_of_sets):

    col = []

    precc = precc_col['PRECC']
    precl = precl_col['PRECL']

    precc.attrs['cell_measures'] = 'area: cell_area'
    precl.attrs['cell_measures'] = 'area: cell_area'

    col.append(precc)
    col.append(precl)

    contU = _preprocess(list_of_sets, col)
    if not contU:
        return

    num_sets = len(list_of_sets[0])

    precc_data = []
    precl_data = []
    for set in list_of_sets[0]:
        precc_data.append(precc.sel(collection=set))
    for set in list_of_sets[1]:
        precl_data.append(precl.sel(collection=set))

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
def cam_ep(qflx_col, precc_col, precl_col, list_of_sets):

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

    contU = _preprocess(list_of_sets, col)
    if not contU:
        return

    num_sets = len(list_of_sets[0])

    qflx_data = []
    precc_data = []
    precl_data = []
    for set in list_of_sets[0]:
        qflx_data.append(qflx.sel(collection=set))
    for set in list_of_sets[1]:
        precc_data.append(precc.sel(collection=set))
    for set in list_of_sets[2]:
        precl_data.append(precl.sel(collection=set))

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
def cam_ressurf(fsns_col, flns_col, shflx_col, lhflx_col, list_of_sets):

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

    contU = _preprocess(list_of_sets, col)
    if not contU:
        return

    num_sets = len(list_of_sets[0])

    fsns_data = []
    flns_data = []
    shflx_data = []
    lhflx_data = []
    for set in list_of_sets[0]:
        fsns_data.append(fsns.sel(collection=set))
    for set in list_of_sets[1]:
        flns_data.append(flns.sel(collection=set))
    for set in list_of_sets[2]:
        shflx_data.append(shflx.sel(collection=set))
    for set in list_of_sets[3]:
        lhflx_data.append(lhflx.sel(collection=set))

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
