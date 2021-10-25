import numpy as np
import xarray as xr


def preprocess(ds, varnames):
    return ds[varnames]


def collect_datasets(data_type, varnames, list_of_ds, labels, **kwargs):
    """
    Concatenate several different xarray datasets across a new
    "collection" dimension, which can be accessed with the specified
    labels.  Stores them in an xarray dataset which can be passed to
    the ldcpy plot functions (Call this OR open_datasets() before
    plotting.)


    Parameters
    ==========
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

    # preprocess
    for i, myds in enumerate(list_of_ds):
        list_of_ds[i] = preprocess(myds, varnames)

    if data_type == 'cam-fv':
        weights_name = 'gw'
        varnames.append(weights_name)
    elif data_type == 'pop':
        weights_name = 'TAREA'
        varnames.append(weights_name)

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

    # full_ds = full_ds.drop(weights_name)

    full_ds['collection'] = xr.DataArray(labels, dims='collection')

    print('dataset size in GB {:0.2f}\n'.format(full_ds.nbytes / 1e9))
    full_ds.attrs['data_type'] = data_type

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
