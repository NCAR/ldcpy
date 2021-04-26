import cf_xarray as cf
import dask
import matplotlib as mpl
import numpy as np
import xarray as xr

# class CAMBudgets:


def cam_restom(fsnt_col, flnt_col, fsnt_sets, flnt_sets):

    fsnt = fsnt_col['FSNT']
    flnt = flnt_col['FLNT']

    num = len(fsnt_sets)
    num2 = len(flnt_sets)

    if num < 2 or num2 < 2 or num != num2:
        print('Must have at least 2 set lables for each variable (and the same number for each).')
        return

    fsnt_data = []
    flnt_data = []
    out_array = np.zeros(num)

    for set in fsnt_sets:
        fsnt_data.append(fsnt.sel(collection=set))
    for set in flnt_sets:
        flnt_data.append(flnt.sel(collection=set))

    for j in range(num):
        tmp = np.mean(fsnt_data[j] - flnt_data[j])
        out_array[j] = tmp

        # need to normalize by area

    print(out_array)


def cam_precip(precc_col, precl_col, precc_sets, precl_sets):

    precc = precc_col['PRECC']
    precl = precl_col['PRECL']

    num = len(precc_sets)
    num2 = len(precl_sets)

    if num < 2 or num2 < 2 or num != num2:
        print('Must have at least 2 set lables for each variable (and the same number for each).')
        return

    precc_data = []
    precl_data = []
    out_array = np.zeros(num)

    for set in precc_sets:
        precc_data.append(precc.sel(collection=set))
    for set in precl_sets:
        precl_data.append(precl.sel(collection=set))

    for j in range(num):
        tmp = np.mean(precc_data[j] + precl_data[j])
        out_array[j] = tmp

        # need to normalize by area

    print(out_array)


def cam_ep(qflx_col, precc_col, precl_col, qflx_sets, precc_sets, precl_sets):

    qflx = qflx_col['QFLX']
    precc = precc_col['PRECC']
    precl = precl_col['PRECL']

    num = len(qflx_sets)
    num2 = len(precc_sets)
    num3 = len(precl_sets)

    if num < 2 or num2 < 2 or num3 < 2 or num != num2 or num2 != num3:
        print('Must have at least 2 set lables for each variable (and the same number for each).')
        return

    # QFLX is "kg/m2/s or mm/s
    # PRECC and PRECL are m/s
    # 1 kg/m2/s = 86400 mm/day.
    # 8.64e7 mm/day = 1 m/sec

    qflx_data = []
    precc_data = []
    precl_data = []
    out_array = np.zeros(num)

    for set in qflx_sets:
        qflx_data.append(qflx.sel(collection=set))
    for set in precc_sets:
        precc_data.append(precc.sel(collection=set))
    for set in precl_sets:
        precl_data.append(precl.sel(collection=set))

    # adjust units

    for j in range(num):
        tmp = np.mean(qflx_data[j] * 86400 + (precc_data[j] + precl_data[j]) * 8.64e7)
        out_array[j] = tmp
        # need to normalize by area

    print(out_array)


def cam_ressurf(
    fsns_col, flns_col, shflx_col, lhflx_col, fsns_sets, flns_sets, shflx_sets, lhflx_sets
):

    # all in W/m^2
    fsns = fsns_col['FSNS']
    flns = flns_col['FLNS']
    shflx = shflx_col['SHFLX']
    lhflx = lhflx_col['LHFLX']

    # maybe read in qflx instead of lhflx and multiply by Lv = 2.501e6
    # to get lhflx? (ncl code does this)

    num = len(fsns_sets)
    num2 = len(flns_sets)
    num3 = len(shflx_sets)
    num4 = len(lhflx_sets)

    if num < 2 or num2 < 2 or num3 < 2 or num4 < 2 or num != num2 or num2 != num3 or num3 != num4:
        print('Must have at least 2 set lables for each variable (and the same number for each).')
        return

    fsns_data = []
    flns_data = []
    shflx_data = []
    lhflx_data = []

    out_array = np.zeros(num)

    for set in fsns_sets:
        fsns_data.append(fsns.sel(collection=set))
    for set in flns_sets:
        flns_data.append(flns.sel(collection=set))
    for set in shflx_sets:
        shflx_data.append(shflx.sel(collection=set))
    for set in lhflx_sets:
        lhflx_data.append(lhflx.sel(collection=set))

    for j in range(num):
        tmp = np.mean(fsns_data[j] - flns_data[j] - shflx_data[j] - lhflx_data[j])
        out_array[j] = tmp

        # need to normalize by area

    print(out_array)
