import numpy as np
import pandas as pd
import xarray as xr

times = pd.date_range('2000-01-01', periods=10)
lats = [0, 1, 2, 3]
lons = [0, 1, 2, 3, 4]
test_data = xr.DataArray(
    np.arange(-100, 100).reshape(4, 5, 10),
    coords=[lats, lons, times],
    dims=['lat', 'lon', 'time'],
)
test_data_2 = xr.DataArray(
    np.arange(-99, 101).reshape(4, 5, 10),
    coords=[lats, lons, times],
    dims=['lat', 'lon', 'time'],
)
