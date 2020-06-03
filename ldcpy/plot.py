import datetime
import math

import cartopy
import cartopy.crs as ccrs
import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss
import xarray as xr
from cartopy.util import add_cyclic_point

import ldcpy.metrics as lm


def compare_plot(ds_o, ds_r, varname, method_str, title_1, title_2, color, nlevs=24):
    lat_o = ds_o['lat']
    lat_r = ds_r['lat']
    cy_data_o, lon_o = add_cyclic_point(ds_o, coord=ds_o['lon'])
    cy_data_r, lon_r = add_cyclic_point(ds_r, coord=ds_r['lon'])
    fig = plt.figure(dpi=300, figsize=(9, 2.5))

    mymap = plt.get_cmap(f'{color}')

    # both plots use same contour levels
    levels = _calc_contour_levels(cy_data_o, cy_data_r, nlevs)

    ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=0.0))
    title_1 = f'orig:{varname}: {title_1}'
    ax1.set_title(title_1)
    pc1 = ax1.contourf(
        lon_o, lat_o, cy_data_o, transform=ccrs.PlateCarree(), cmap=mymap, levels=levels
    )
    ax1.set_global()
    ax1.coastlines()

    ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=0.0))
    title_2 = f'{method_str}:{varname}: {title_2}'
    ax2.set_title(title_2)
    ax2.contourf(lon_r, lat_r, cy_data_r, transform=ccrs.PlateCarree(), cmap=mymap, levels=levels)
    ax2.set_global()
    ax2.coastlines()

    # add colorbar
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
    cax = fig.add_axes([0.1, 0, 0.8, 0.05])
    cbar = fig.colorbar(pc1, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8, rotation=30)


def plot(name, ds, varname, method_str):
    """
    visualize the mean error
    want to be able to input multiple?
    """

    lat = ds['lat']
    cy_data, lon = add_cyclic_point(ds, coord=ds['lon'])

    mymap = plt.get_cmap('coolwarm')

    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0.0))
    pc = ax.pcolormesh(lon, lat, cy_data, transform=ccrs.PlateCarree(), cmap=mymap)
    #    pc = ax.contourf(lon, lat, cy_data, transform=ccrs.PlateCarree(), cmap=mymap, levels=nlevs)
    cb = plt.colorbar(pc, orientation='horizontal', shrink=0.95)
    cb.ax.tick_params(labelsize=8, rotation=30)

    ax.set_global()
    ax.coastlines()
    title = f'{varname} ({method_str}): {name}'
    ax.set_title(title)


###############


def time_series_plot(ds, varname, method_str, resolution='dayofyear', plot_type='normal'):
    """
    error time series
    """
    if resolution == 'dayofyear':
        tick_interval = 20
        xlabel = 'day of year'
    elif resolution == 'month':
        tick_interval = 1
        xlabel = 'month'
    elif resolution == 'year':
        tick_interval = 1
        xlabel = 'year'

    if plot_type == 'normal':
        plot_data = ds
        plot_ylabel = 'error'
    elif plot_type == 'log':
        plot_data = xr.ufuncs.log10(ds)
        plot_ylabel = 'log10(error)'

    mpl.pyplot.plot(plot_data[resolution].data, plot_data)
    mpl.pyplot.ylabel(plot_ylabel)
    mpl.pyplot.xlabel(xlabel)
    mpl.pyplot.xticks(
        np.arange(min(plot_data[resolution]), max(plot_data[resolution]) + 1, tick_interval)
    )
    mpl.pyplot.title(f'{varname} ({method_str}): Mean Absolute Error by {xlabel.capitalize()}')


###############


def _calc_contour_levels(dat_1, dat_2, nlevs):
    """
    TODO: minval returns the smallest value not equal to -inf, is there a more elegant solution to plotting -inf values
    (for EW contrast variance in particular)?
    """
    # both plots use same contour levels
    minval = np.nanmin(np.minimum(dat_1, dat_2))
    if minval == -math.inf:
        minval = np.minimum(dat_1[np.isfinite(dat_1)].min(), dat_2[np.isfinite(dat_2)].min())
    maxval = np.nanmax(np.maximum(dat_1, dat_2))
    if maxval == math.inf:
        maxval = np.maximum(dat_1[np.isfinite(dat_1)].max(), dat_2[np.isfinite(dat_2)].max())
    levels = minval + np.arange(nlevs + 1) * (maxval - minval) / nlevs
    # print('Min value: {}\nMax value: {}'.format(minval, maxval))
    return levels
