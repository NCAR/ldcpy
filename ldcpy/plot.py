import numpy as np
import matplotlib.pyplot as plt
#import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs
import cmocean
from cartopy.util import add_cyclic_point

###############

def compare_mean(ds, varname, ens_o, ens_r, method_str, nlevs=24):
    """
    visualize mean value at each grid point for orig and compressed (time-series)
    assuming FV data and put the weighted mean
    """
    mean_data_o = ds[varname].sel(ensemble=ens_o).mean(dim='time')
    mean_data_r = ds[varname].sel(ensemble=ens_r).mean(dim='time')

    #weighted mean
    gw = ds['gw'].values
    o_wt_mean = np.average(np.average(mean_data_o,axis=0, weights=gw))
    r_wt_mean = np.average(np.average(mean_data_r,axis=0, weights=gw))

    lat = ds['lat']
    cy_data_o, lon_o = add_cyclic_point(mean_data_o, coord=ds['lon'])
    cy_data_r, lon_r = add_cyclic_point(mean_data_r, coord=ds['lon'])
    fig = plt.figure(dpi=300, figsize=(9, 2.5))

    mymap = cmocean.cm.thermal
    # both plots use same contour levels
    levels = _calc_contour_levels(cy_data_o, cy_data_r, nlevs)

    ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=0.0))
    title = f'orig:{varname} : mean = {o_wt_mean:.2f}'
    ax1.set_title(title)
    pc1 = ax1.contourf(lon_o, lat, cy_data_o, transform=ccrs.PlateCarree(), cmap=mymap, levels=levels)
    ax1.set_global()
    ax1.coastlines()

    ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=0.0))
    title = f'{method_str}:{varname} : mean = {r_wt_mean:.2f}'
    ax2.set_title(title)
    pc2 = ax2.contourf(lon_r, lat, cy_data_r, transform=ccrs.PlateCarree(), cmap=mymap, levels=levels)
    ax2.set_global()
    ax2.coastlines()

    # add colorbar
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
    cax = fig.add_axes([0.1, 0, 0.8, 0.05])
    cbar = fig.colorbar(pc1, cax=cax,  orientation='horizontal')
    cbar.ax.tick_params(labelsize=8, rotation=30)

###############

def compare_std(ds, varname, ens_o, ens_r, method_str, nlevs=24):
    """
    TODO: visualize std dev at each grid point for orig and compressed (time-series)
    assuming FV mean
    """
    std_data_o = ds[varname].sel(ensemble=ens_o).std(dim='time', ddof=1)
    std_data_r = ds[varname].sel(ensemble=ens_r).std(dim='time', ddof=1)

    lat = ds['lat']
    cy_data_o, lon_o = add_cyclic_point(std_data_o, coord=ds['lon'])
    cy_data_r, lon_r = add_cyclic_point(std_data_r, coord=ds['lon'])
    fig = plt.figure(dpi=300, figsize=(9, 2.5))

    mymap = plt.get_cmap('coolwarm')
    # both plots use same contour levels
    levels = _calc_contour_levels(cy_data_o, cy_data_r, nlevs)

    ax1 = plt.subplot(1, 2, 1, projection=ccrs.Robinson(central_longitude=0.0))
    title = f'orig:{varname}: std'
    ax1.set_title(title)
    pc1 = ax1.contourf(lon_o, lat, cy_data_o, transform=ccrs.PlateCarree(), cmap=mymap, levels=levels)
    ax1.set_global()
    ax1.coastlines()

    ax2 = plt.subplot(1, 2, 2, projection=ccrs.Robinson(central_longitude=0.0))
    title = f'{method_str}:{varname}: std'
    ax2.set_title(title)
    pc2 = ax2.contourf(lon_r, lat, cy_data_r, transform=ccrs.PlateCarree(), cmap=mymap, levels=levels)
    ax2.set_global()
    ax2.coastlines()

    # add colorbar
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95)
    cax = fig.add_axes([0.1, 0, 0.8, 0.05])
    cbar = fig.colorbar(pc1, cax=cax,  orientation='horizontal')
    cbar.ax.tick_params(labelsize=8, rotation=30)

###############

def mean_error(ds, varname, ens_o, ens_r, method_str):
    """
    visualize the mean error
    want to be able to input multiple?
    """
    e = ds[varname].sel(ensemble=ens_o) - ds[varname].sel(ensemble=ens_r)
    mean_e = e.mean(dim='time')
    lat = ds['lat']
    cy_data, lon = add_cyclic_point(mean_e, coord=ds['lon'])
    myfig = plt.figure(dpi=300)

    mymap = plt.get_cmap('coolwarm')
    nlevs = 24

    ax = plt.subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0.0))
    pc = ax.pcolormesh(lon, lat, cy_data, transform=ccrs.PlateCarree(), cmap=mymap)
#    pc = ax.contourf(lon, lat, cy_data, transform=ccrs.PlateCarree(), cmap=mymap, levels=nlevs)
    cb = plt.colorbar(pc, orientation='horizontal', shrink=.95)
    cb.ax.tick_params(labelsize=8, rotation=30)

    ax.set_global()
    ax.coastlines()
    title = f'{varname} ({method_str}): mean error '
    ax.set_title(title)

###############

def error_time_series(ds, varname, ens_o, ens_r):
    """
    error time series
    """
    pass

###############

def _calc_contour_levels(dat_1, dat_2, nlevs):
    # both plots use same contour levels
    minval = np.floor(np.nanmin(np.minimum(dat_1, dat_2)))
    maxval = np.ceil(np.nanmax(np.maximum(dat_1, dat_2)))
    levels = minval + np.arange(nlevs+1)*(maxval - minval)/nlevs
    #print('Min value: {}\nMax value: {}'.format(minval, maxval))
    return levels