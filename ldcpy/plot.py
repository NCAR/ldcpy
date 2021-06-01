import calendar
import copy
import warnings

import cartopy
import cf_xarray as cf
import cmocean
import matplotlib as mpl
import numpy as np
import xarray as xr
import xrft
from cartopy import crs as ccrs
from cartopy.util import add_cyclic_point
from matplotlib import pyplot as plt
from pylab import flipud

from ldcpy import calcs as lm, util as lu
from ldcpy.convert import CalendarDateTime

xr.set_options(keep_attrs=True)


def tex_escape(text):
    """
    :param text: a plain text message
    :return: the message escaped to appear correctly in LaTeX
    """
    # conv = {
    #    '&': r'\&',
    #    '%': r'\%',
    #    '$': r'\$',
    #    '#': r'\#',
    #    '_': r'\_',
    #    '{': r'\{',
    #    '}': r'\}',
    #    '^': r'\^{}',
    #    '\\': r'\textbackslash{}',
    #    '<': r'\textless{}',
    #    '>': r'\textgreater{}',
    # }
    # regex = re.compile(
    #    '|'.join(re.escape(str(key)) for key in sorted(conv.keys(), key=lambda item: -len(item)))
    # )
    # return regex.sub(lambda match: conv[match.group()], text)
    return text


class calcsPlot(object):
    """
    This class contains code to plot calcs in an xarray Dataset that has either 'lat' and 'lon' dimensions, or a
    'time' dimension.
    """

    def __init__(
        self,
        ds,
        varname,
        calc,
        sets,
        group_by=None,
        scale='linear',
        calc_type='raw',
        plot_type='spatial',
        transform='none',
        subset=None,
        approx_lat=None,
        approx_lon=None,
        lev=0,
        color='coolwarm',
        standardized_err=False,
        quantile=None,
        calc_ssim=False,
        contour_levs=24,
        short_title=False,
        axes_symmetric=False,
        legend_loc='upper right',
        vert_plot=False,
        tex_format=False,
        legend_offset=None,
        weighted=True,
    ):

        self._ds = ds

        # calc settings used in plot titles
        self._varname = varname
        self._sets = sets
        self._title_lat = None
        self._title_lon = None

        # Plot settings
        self._calc = calc
        self._group_by = group_by
        self._scale = scale
        self._calc_type = calc_type
        self._plot_type = plot_type
        self._subset = subset
        self._true_lat = approx_lat
        self._true_lon = approx_lon
        self._transform = transform
        self._lev = lev
        self._color = color
        self._short_title = short_title
        self._quantile = quantile
        self._calc_ssim = calc_ssim
        self._contour_levs = contour_levs
        self._axes_symmetric = axes_symmetric
        self._legend_loc = legend_loc
        self.vert_plot = vert_plot
        self._tex_format = tex_format
        self._legend_offset = legend_offset
        self._weighted = weighted

    def verify_plot_parameters(self):
        if len(self._sets) < 2 and self._calc_type in [
            'diff',
            'ratio',
            'calc_of_diff',
        ]:
            raise ValueError(f'Must specify set2 for {self._calc_type} calc type')
        if self._plot_type in ['spatial'] and self._group_by is not None:
            raise ValueError(f'Cannot group by {self._group_by} in a non-time-series plot')
        if self._plot_type not in ['spatial'] and self._color != 'coolwarm':
            raise ValueError('Cannot change color scheme in a non-spatial plot')
        if self._plot_type in ['spatial'] and (
            self._true_lat is not None or self._true_lon is not None
        ):
            raise ValueError('Cannot currently subset by latitude or longitude in a spatial plot')
        if self._lev != 0:  # and 'lev' not in self._ds.dims:
            try:
                vert = self._ds.cf['vertical']
            except KeyError:
                vert = None
            if vert is None:
                raise ValueError('Cannot subset by lev (vertical dimension) in this dataset')
        if self._quantile is not None and self._calc != 'quantile':
            raise ValueError('Cannot change quantile value if calc is not quantile')
        if self._quantile is None and self._calc == 'quantile':
            raise ValueError('Must specify quantile value as argument')

        if self._calc in ['lag1', 'corr_lag1', 'mae_day_max'] and self._plot_type not in [
            'spatial',
        ]:
            raise ValueError(f'Cannot plot {self._calc} in a non-spatial plot')

    def get_calcs(self, da):
        da_data = da
        da_data.attrs = da.attrs

        # lat/lon dim names are different for ocn and atm
        dd = da_data.cf[da_data.cf.coordinates['latitude'][0]].dims

        ll = len(dd)
        if ll == 1:
            lat_dim = dd[0]
            lon_dim = da_data.cf['longitude'].dims[0]
        elif ll == 2:
            lat_dim = dd[0]
            lon_dim = dd[1]

        if self._plot_type in ['spatial']:
            calcs_da = lm.Datasetcalcs(da_data, ['time'], weighted=self._weighted)
        elif self._plot_type in ['time_series', 'periodogram', 'histogram']:
            calcs_da = lm.Datasetcalcs(da_data, [lat_dim, lon_dim], weighted=self._weighted)
        else:
            raise ValueError(f'plot type {self._plot_type} not supported')

        if self._calc_ssim and self._plot_type != 'spatial':
            warnings.warn(
                'SSIM is only calculated for spatial plots, ignoring calc_ssim option', UserWarning
            )

        raw_data = calcs_da.get_calc(self._calc, self._quantile, self._group_by)

        return raw_data

    def get_plot_data(self, raw_data_1, raw_data_2=None):
        if self._calc_type == 'diff':
            plot_data = raw_data_1 - raw_data_2
            plot_data.attrs = raw_data_1.attrs
        elif self._calc_type == 'ratio':
            plot_data = raw_data_2 / raw_data_1
            plot_data.attrs = raw_data_1.attrs
            if hasattr(self._ds, 'units'):
                self._odds_positive.attrs['units'] = ''

        elif self._calc_type == 'raw' or self._calc_type == 'calc_of_diff':
            plot_data = raw_data_1
        else:
            raise ValueError(f'calc_type {self._calc_type} not supported')

        if self._group_by is not None and self._calc not in [
            'standardized_mean',
            'odds_positive',
        ]:
            plot_attrs = plot_data.attrs
            plot_data = plot_data.groupby(self._group_by).mean(dim='time')
            plot_data.attrs = plot_attrs

        if self._transform == 'none':
            pass
        elif self._transform == 'log':
            plot_attrs = plot_data.attrs
            plot_data = np.log10(plot_data)
            plot_data.attrs = plot_attrs
        else:
            raise ValueError(f'calc transformation {self._transform} not supported')

        return plot_data

    def get_title(self, calc_name, c_name=None):

        if c_name is not None:
            das = f'{c_name}'
        else:
            das = f'{self._sets[0]}'

        if self._short_title is True:
            if self._plot_type == 'time_series':
                return ''
            else:
                return das

        if self._quantile is not None and calc_name == 'quantile':
            calc_full_name = f'{calc_name} {self._quantile}'
        else:
            calc_full_name = calc_name

        if self._transform == 'log':
            title = f'{self._varname}: log10 {calc_full_name}'
        else:
            title = f'{self._varname}: {calc_full_name}'

        if self._plot_type == 'spatial':
            title = f'{das}: {title}'

        if self._calc_type != 'raw':
            title = f'{title} {self._calc_type}'

        if self._group_by is not None:

            title = f'{title} by {self._group_by}'

        if self.title_lat is not None:
            if self.title_lon is not None:
                title = f'{title} at lat={self.title_lat:.2f}, lon={self.title_lon:.2f}'
            else:
                title = f'{title} at lat={self.title_lat:.2f}'
        elif self.title_lon is not None:
            title = f'{title} at lat={self.title_lon:.2f}'

        if self._subset is not None:
            title = f'{title} subset:{self._subset}'

        if self._plot_type == 'histogram':
            title = f'time-series histogram:{title}'
        elif self._plot_type == 'periodogram':
            title = f'periodogram:{title}'

        return title

    def _label_offset(
        self,
        ax,
    ):
        fmt = ax.yaxis.get_major_formatter()
        ax.yaxis.offsetText.set_visible(False)
        set_label = ax.set_ylabel
        label = ax.get_ylabel()

        def update_label(event_axes):
            offset = fmt.get_offset()
            if offset == '':
                set_label('{}'.format(label))
            else:
                set_label('{} ({})'.format(label, offset))
            return

        ax.callbacks.connect('ylim_changed', update_label)
        ax.figure.canvas.draw()
        update_label(None)
        return

    def spatial_plot(self, da_sets, titles):

        if self.vert_plot:
            nrows = int((da_sets.sets.size))
        else:
            nrows = int((da_sets.sets.size + 1) / 2)
        if len(da_sets) == 1:
            ncols = 1
        else:
            ncols = 2
        if self._calc == 'zscore':
            ncols = 1
            nrows = len(da_sets)

        if self.vert_plot:
            fig = plt.figure(dpi=300, figsize=(4.5, 2.5 * nrows))
            plt.rcParams.update({'font.size': 10})
        else:
            fig = plt.figure(dpi=300, figsize=(9, 2.5 * nrows))
            plt.rcParams.update({'font.size': 10})

        mymap = copy.copy(mpl.cm.get_cmap(f'{self._color}'))
        mymap.set_under(color='black')
        mymap.set_over(color='white')
        mymap.set_bad(alpha=0)

        axs = {}
        psets = {}
        nan_inf_flag = 0
        all_nan_flag = 0

        cmax = []
        cmin = []

        # lat/lon could be 1 or 2d and have different names
        lon_coord_name = da_sets[0].cf.coordinates['longitude'][0]
        lat_coord_name = da_sets[0].cf.coordinates['latitude'][0]

        # is the lat/lon 1d or 2d (to do: set error if > 2)
        latdim = da_sets[0].cf[lon_coord_name].ndim

        central = 0.0  # might make this a parameter later
        if latdim == 2:  # probably pop
            central = 300.0

        for i in range(da_sets.sets.size):

            if self.vert_plot:
                axs[i] = plt.subplot(
                    nrows, 1, i + 1, projection=ccrs.Robinson(central_longitude=central)
                )
            else:
                axs[i] = plt.subplot(
                    nrows, ncols, i + 1, projection=ccrs.Robinson(central_longitude=central)
                )

            axs[i].set_facecolor('#39ff14')

            # make data periodic
            if latdim == 2:
                ylon = da_sets[i][lon_coord_name]
                lon_sets = np.hstack((ylon, ylon[:, 0:1]))

                xlat = da_sets[i][lat_coord_name]
                lat_sets = np.hstack((xlat, xlat[:, 0:1]))

                cy_datas = add_cyclic_point(da_sets[i])
            else:  # 1d
                ylon = da_sets[i][lon_coord_name]
                lon_sets = np.hstack((ylon, ylon[0]))
                lat_sets = da_sets[i][lat_coord_name]

                cy_datas = add_cyclic_point(da_sets[i])

            if np.isnan(cy_datas).any() or np.isinf(cy_datas).any():
                nan_inf_flag = 1
            if np.isnan(cy_datas).all():
                all_nan_flag = 1

            cyxr = xr.DataArray(data=cy_datas)

            if not np.isinf(cyxr).all():
                cmin.append(np.min(cyxr.where(cyxr != -np.inf).min()))
                cmax.append(np.max(cyxr.where(cyxr != np.inf).max()))

            if latdim == 2:
                no_inf_data_set = np.nan_to_num(cyxr.astype(np.float32), nan=np.nan)
            else:
                ncyxr = cyxr.roll(dim_1=145)
                no_inf_data_set = np.nan_to_num(ncyxr.astype(np.float32), nan=np.nan)

            # casting to float32 from float64 using imshow prevents lots of tiny black dots from showing up in some plots with lots of
            # zeroes. See plot of probability of negative PRECT to see this in action.
            if latdim == 2:
                psets[i] = psets[i] = axs[i].pcolormesh(
                    lon_sets,
                    lat_sets,
                    no_inf_data_set,
                    transform=ccrs.PlateCarree(),
                    cmap=mymap,
                )
            else:
                psets[i] = axs[i].imshow(
                    img=flipud(no_inf_data_set), transform=ccrs.PlateCarree(), cmap=mymap
                )

            # psets[i] = axs[i].imshow(
            #    img=flipud(no_inf_data_set), transform=ccrs.PlateCarree(), cmap=mymap
            # )
            axs[i].set_global()

            # if we want to get the ssim
            if self._calc_ssim:
                axs[i].axis('off')
                plt.margins(0, 0)
                extent1 = axs[i].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                axs[i].imshow
                plt.savefig(f'tmp_ssim{i+1}', bbox_inches=extent1, transparent=True, pad_inches=0)
                axs[i].axis('on')

            # may need to be modified for other components
            if latdim == 1:
                axs[i].coastlines()
            else:
                axs[i].add_feature(
                    cartopy.feature.NaturalEarthFeature(
                        'physical',
                        'land',
                        '110m',
                        linewidth=0.5,
                        edgecolor='black',
                        facecolor='darkgray',
                    )
                )

            axs[i].set_title(tex_escape(titles[i]))
            del cy_datas

            # end of for loopon plots

        if len(cmin) > 0:
            color_min = np.min(cmin)
        else:
            color_min = -0.1
        if len(cmax) > 0:
            color_max = np.max(cmax)
        else:
            color_max = 0.1

        if self._axes_symmetric:
            color_max_abs = max(abs(color_min), abs(color_max))
            color_min = -1 * color_max_abs
            color_max = color_max_abs
        for i in range(len(psets)):
            psets[i].set_clim(color_min, color_max)
            pass

        # add colorbar
        if self.vert_plot is False:
            fig.subplots_adjust(left=0.1, right=0.9, bottom=0.2, top=0.95)

        cbs = []
        if not all_nan_flag:
            cax = fig.add_axes([0.1, 0, 0.8, 0.05])

            for i in range(len(psets)):
                cbs.append(fig.colorbar(psets[i], cax=cax, orientation='horizontal', shrink=0.95))
                cbs[i].ax.set_title(f'{da_sets[i].units}')
                if self.vert_plot:
                    cbs[i].ax.set_aspect(0.03)
                    cbs[i].ax.set_anchor((0, 1.35 + 0.15 * (nrows - 1)))
                else:
                    cbs[i].ax.set_aspect(0.03)
                    if len(psets) > 2:
                        cbs[i].ax.set_anchor((0, 1.35 + 0.15 * (nrows - 1)))
                    else:
                        cbs[i].ax.set_anchor((0.5, 1.35 + 0.15 * (nrows - 1)))
                cbs[i].ax.tick_params(labelsize=8, rotation=30)
            if nan_inf_flag:
                proxy = [
                    plt.Rectangle((0, 0), 1, 1, fc='#39ff14'),
                    plt.Rectangle((0, 1), 2, 2, fc='#000000'),
                    plt.Rectangle((0, 1), 2, 2, fc='#ffffff', edgecolor='black'),
                ]
                if self.vert_plot:
                    plt.rcParams.update({'font.size': 8})
                    plt.legend(
                        proxy,
                        ['NaN', '-Inf', 'Inf'],
                        loc='lower center',
                        bbox_to_anchor=(0.51, -6),
                        ncol=len(proxy),
                    )
                else:
                    plt.rcParams.update({'font.size': 10})
                    if len(psets) > 2:
                        plt.legend(
                            proxy,
                            ['NaN', '-Inf', 'Inf'],
                            bbox_to_anchor=(0.672, 4),
                            ncol=len(proxy),
                        )
                    else:
                        plt.legend(
                            proxy,
                            ['NaN', '-Inf', 'Inf'],
                            bbox_to_anchor=(0.78, -2),
                            ncol=len(proxy),
                        )
        else:
            fig.add_axes([0.1, 0, 0.8, 0.05])
            proxy = [
                plt.Rectangle((0, 0), 1, 1, fc='#39ff14'),
                plt.Rectangle((0, 1), 2, 2, fc='#000000'),
                plt.Rectangle((0, 1), 2, 2, fc='#ffffff', edgecolor='black'),
            ]
            plt.legend(proxy, ['NaN', '-Inf', 'Inf'], bbox_to_anchor=(0.87, 2), ncol=len(proxy))
            plt.axis('off')

        if self._calc_ssim:
            import os

            import skimage.io
            from skimage.metrics import structural_similarity as ssim

            for i in range(1, len(da_sets)):
                img1 = skimage.io.imread('tmp_ssim1.png')
                img2 = skimage.io.imread(f'tmp_ssim{i+1}.png')
                # ssim_val = ssim(img1, img2, multichannel=True)
                ssim_val = ssim(
                    img1,
                    img2,
                    multichannel=True,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                )
                print(f' SSIM 1 & {i+1} = % 5.5f\n' % (ssim_val))
            for i in range(len(da_sets) + 1):
                if os.path.exists(f'tmp_ssim{i}.png'):
                    os.remove(f'tmp_ssim{i}.png')

    def hist_plot(self, plot_data, title):
        fig, axs = mpl.pyplot.subplots(1, 1, sharey=True, tight_layout=True)
        sets = []
        for set in plot_data.sets:
            sets.append(plot_data.sel(sets=set))
        axs.hist(sets, label=plot_data.sets.data)
        if plot_data.units != '':
            mpl.pyplot.xlabel(tex_escape(f'{self._calc} ({plot_data.units})'))
        else:
            mpl.pyplot.xlabel(tex_escape(f'{self._calc}'))
        mpl.pyplot.title(tex_escape(title[0]))
        if self.vert_plot:
            plt.legend(loc=self._legend_loc, borderaxespad=1.0)
            plt.rcParams.update({'font.size': 16})
        else:
            plt.rcParams.update({'font.size': 10})
            if self._legend_loc is None:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=self._legend_loc, borderaxespad=0.0)
            else:
                plt.legend(
                    bbox_to_anchor=self._legend_offset, loc=self._legend_loc, borderaxespad=0.0
                )

    def periodogram_plot(self, plot_data, title):
        plt.figure()
        for j in range(plot_data.sets.size):
            dat = xrft.dft(
                xr.DataArray(plot_data[j].data - plot_data[j].data.mean()).chunk(
                    (plot_data[j].data - plot_data[j].data.mean()).size
                )
            )
            i = (np.multiply(dat, np.conj(dat)) / dat.size).real
            i = np.log10(i[2 : int(dat.size / 2) + 1])
            freqs = np.array(range(1, int(dat.size / 2))) / dat.size

            mpl.pyplot.plot(freqs, i, label=plot_data[j].sets.data)
        if self.vert_plot:
            plt.legend(loc=self._legend_loc, borderaxespad=1.0)
            plt.rcParams.update({'font.size': 16})
        else:
            plt.rcParams.update({'font.size': 10})
            if self._legend_loc is None:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=self._legend_loc, borderaxespad=0.0)
            else:
                plt.legend(
                    bbox_to_anchor=self._legend_offset, loc=self._legend_loc, borderaxespad=0.0
                )

        mpl.pyplot.title(tex_escape(title[0]))
        mpl.pyplot.ylabel('Spectrum')
        mpl.pyplot.xlabel('Frequency')

    def time_series_plot(
        self,
        da_sets,
        titles,
    ):
        """
        time series plot
        """

        group_string = 'time.year'
        xlabel = 'date'
        tick_interval = int(da_sets.size / da_sets.sets.size / 5) + 1
        if da_sets.size / da_sets.sets.size == 1:
            tick_interval = 1
        if self._group_by == 'time.dayofyear':

            group_string = 'dayofyear'
            xlabel = 'Day of Year'
        elif self._group_by == 'time.month':
            group_string = 'month'
            xlabel = 'Month'
            tick_interval = 1
        elif self._group_by == 'time.year':
            group_string = 'year'
            xlabel = 'Year'
        elif self._group_by == 'time.day':
            group_string = 'day'
            xlabel = 'Day'

        if self._calc_type == 'diff':
            if da_sets.units != '':
                ylabel = f'{self._calc} ({da_sets.units}) diff'
            else:
                ylabel = f'{self._calc} diff'
        elif self._calc_type == 'ratio':
            ylabel = f'{self._calc} ratio'
        elif self._calc_type == 'calc_of_diff':
            if da_sets.units != '':
                ylabel = f'{self._calc} ({da_sets.units}) of diff'
            else:
                ylabel = f'{self._calc} of diff'
        else:
            if da_sets.units != '':
                ylabel = f'{self._calc} ({da_sets.units})'
            else:
                ylabel = f'{self._calc}'

        if self._transform == 'log':
            plot_ylabel = f'log10 {ylabel}'
        else:
            plot_ylabel = ylabel

        mpl.style.use('default')

        plt.figure()
        if self.vert_plot:
            plt.rcParams.update({'font.size': 16})
        else:
            plt.rcParams.update({'font.size': 10})
        plt.rcParams.update(
            {
                'text.usetex': self._tex_format,
            }
        )

        for i in range(da_sets.sets.size):
            if self._group_by is not None:
                plt.plot(
                    da_sets[i][group_string].data,
                    da_sets[i],
                    f'C{i}',
                    label=f'{da_sets.sets.data[i]}',
                )
                ax = plt.gca()
            else:
                dtindex = da_sets[i].indexes['time']
                c_d_time = [CalendarDateTime(item, '365_day') for item in dtindex]
                mpl.pyplot.plot(c_d_time, da_sets[i], f'C{i}', label=f'{da_sets.sets.data[i]}')
                ax = plt.gca()
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment('right')

        if self.vert_plot:
            if self._legend_offset is None:
                plt.legend(loc=self._legend_loc, borderaxespad=1.0)
            else:
                plt.legend(
                    loc=self._legend_loc, borderaxespad=1.0, bbox_to_anchor=self._legend_offset
                )
        else:
            plt.rcParams.update({'font.size': 10})
            if self._legend_offset is None:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.0)
            else:
                plt.legend(bbox_to_anchor=self._legend_offset, loc='upper left', borderaxespad=0.0)
        mpl.pyplot.ylabel(tex_escape(plot_ylabel))
        mpl.pyplot.yscale(self._scale)
        self._label_offset(ax)
        mpl.pyplot.xlabel(tex_escape(xlabel))

        if self._group_by is not None:
            mpl.pyplot.xticks(
                np.arange(min(da_sets[group_string]), max(da_sets[group_string]) + 1, tick_interval)
            )
            if self._group_by == 'time.month':
                int_labels = plt.xticks()[0]
                month_labels = [
                    calendar.month_name[i] for i in int_labels if calendar.month_name[i] != ''
                ]
                unique_month_labels = list(dict.fromkeys(month_labels))
                plt.gca().set_xticklabels(unique_month_labels)
                for label in ax.get_xticklabels():
                    label.set_rotation(30)
                    label.set_horizontalalignment('right')
        # else:
        #    mpl.pyplot.xticks(
        #        pd.date_range(
        #            np.datetime64(da_sets['time'].data[0]),
        #            np.datetime64(da_sets['time'].data[-1]),
        #            periods=int(da_sets['time'].size / tick_interval) + 1,
        #        )
        #    )

        mpl.pyplot.title(tex_escape(titles[0]))

    def get_calc_label(self, calc, data):
        dd = data.cf[data.cf.coordinates['latitude'][0]].dims

        ll = len(dd)
        if ll == 1:
            lat_dim = dd[0]
            lon_dim = data.cf['longitude'].dims[0]
        elif ll == 2:
            lat_dim = dd[0]
            lon_dim = dd[1]

        # Get special calc names
        if self._short_title is False:
            if calc == 'zscore':
                zscore_cutoff = lm.Datasetcalcs(
                    (data), ['time'], weighted=self._weighted
                ).get_single_calc('zscore_cutoff')
                percent_sig = lm.Datasetcalcs(
                    (data), ['time'], weighted=self._weighted
                ).get_single_calc('zscore_percent_significant')
                calc_name = f'{calc}: cutoff {zscore_cutoff[0]:.2f}, % sig: {percent_sig:.2f}'
            elif calc == 'mean' and self._plot_type == 'spatial' and self._calc_type == 'raw':

                if self._weighted:
                    a1_data = (
                        lm.Datasetcalcs(data, ['time'], weighted=self._weighted)
                        .get_calc(calc)
                        .cf.weighted('area')
                        .mean()
                        .data.compute()
                    )
                else:
                    a1_data = (
                        lm.Datasetcalcs(data, ['time'], weighted=self._weighted)
                        .get_calc(calc)
                        .mean()
                        .data.compute()
                    )
                    print(a1_data)
                # check for NANs
                # indices = ~np.isnan(a1_data)
                # if weights is not None:
                #    weights = weights[indices]

                # a2_data = np.average(
                #    a1_data[indices],
                #    axis=0,
                #    weights=weights,
                # ).compute()

                # o_wt_mean = np.nanmean(a2_data)

                calc_name = f'{calc} = {a1_data:.2f}'
            elif calc == 'pooled_var_ratio':
                pooled_sd = np.sqrt(
                    lm.Datasetcalcs((data), ['time'], weighted=self._weighted).get_single_calc(
                        'pooled_variance'
                    )
                )
                d = pooled_sd.data.compute()
                calc_name = f'{calc}: pooled SD = {d:.2f}'
            elif calc == 'ann_harmonic_ratio':
                p = lm.Datasetcalcs((data), ['time'], weighted=self._weighted).get_single_calc(
                    'annual_harmonic_relative_ratio_pct_sig'
                )
                calc_name = f'{calc}: % sig = {p:.2f}'
            elif self._plot_type == 'spatial':
                if self._weighted:
                    a1_data = (
                        lm.Datasetcalcs(data, ['time'], weighted=self._weighted)
                        .get_calc(calc)
                        .cf.weighted('area')
                        .mean()
                        .data.compute()
                    )
                else:
                    a1_data = (
                        lm.Datasetcalcs(data, ['time'], weighted=self._weighted)
                        .get_calc(calc)
                        .mean()
                        .data.compute()
                    )

                calc_name = f'{calc} = {a1_data:.2f}'
            elif self._plot_type == 'time_series':
                if self._weighted:
                    a1_data = (
                        lm.Datasetcalcs(data, [lat_dim, lon_dim], weighted=self._weighted)
                        .get_calc(calc)
                        .mean()
                        .data.compute()
                    )
                else:
                    a1_data = (
                        lm.Datasetcalcs(data, [lat_dim, lon_dim], weighted=self._weighted)
                        .get_calc(calc)
                        .mean()
                        .data.compute()
                    )

                calc_name = f'{calc} = {a1_data:.2f}'
            else:
                calc_name = calc

            return calc_name
        else:
            return ''


def plot(
    ds,
    varname,
    calc,
    sets,
    group_by=None,
    scale='linear',
    calc_type='raw',
    plot_type='spatial',
    transform='none',
    subset=None,
    lat=None,
    lon=None,
    lev=0,
    color='coolwarm',
    quantile=None,
    start=None,
    end=None,
    calc_ssim=False,
    short_title=False,
    axes_symmetric=False,
    legend_loc='upper right',
    vert_plot=False,
    tex_format=False,
    legend_offset=None,
    weighted=True,
):
    """
    Plots the data given an xarray dataset


    Parameters
    ==========
    ds : xarray.Dataset
        The input dataset
    varname : str
        The name of the variable to be plotted
    calc : str
        The name of the calc to be plotted (must match a property name in the Datasetcalcs
        class in ldcpy.plot, for more information about the available calcs see ldcpy.Datasetcalcs)
        Acceptable values include:

            - ns_con_var
            - ew_con_var
            - mean
            - std
            - variance
            - prob_positive
            - prob_negative
            - odds_positive
            - zscore
            - mean_abs
            - mean_squared
            - rms
            - sum
            - sum_squared
            - corr_lag1
            - quantile
            - lag1
            - standardized_mean
            - ann_harmonic_ratio
            - pooled_variance_ratio

    sets : list <str>
        The labels of the dataset to gather calcs from
    group_by : str
        how to group the data in time series plots.
        Valid groupings:

            - time.day
            - time.dayofyear
            - time.month
            - time.year
    scale : str, optional
        time-series y-axis plot transformation. (default "linear")
        Valid options:

            - linear
            - log
    calc_type : str, optional
        The type of operation to be performed on the calcs. (default 'raw')
        Valid options:

            - raw: the unaltered calc values
            - diff: the difference between the calc values in the first set and every other set
            - ratio: the ratio of the calc values in (2nd, 3rd, 4th... sets/1st set)
            - calc_of_diff: the calc value computed on the difference between the first set and every other set
    plot_type : str , optional
        The type of plot to be created. (default 'spatial')
        Valid options:

            - spatial: a plot of the world with values at each lat and lon point (takes the mean across the time dimension)
            - time-series: A time-series plot of the data (computed by taking the mean across the lat and lon dimensions)
            - histogram: A histogram of the time-series data
    transform : str, optional
        data transformation. (default 'none')
        Valid options:

            - none
            - log
    subset : str, optional
        subset of the data to gather calcs on (default None).
        Valid options:

            - first5: the first 5 days of data
            - DJF: data from the months December, January, February
            - MAM: data from the months March, April, May
            - JJA: data from the months June, July, August
            - SON: data from the months September, October, November
    lat : float, optional
        The latitude of the data to gather calcs on (default None).
    lon : float , optional
        The longitude of the data to gather calcs on (default None).
    lev : float, optional
        The level of the data to gather calcs on (used if plotting from a 3d data set),
        (default 0).
    color : str, optional
        The color scheme for spatial plots, (default 'coolwarm').
        see https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        for more options
    quantile : float, optional
        A value between 0 and 1 required if calc="quantile", corresponding to the desired quantile to gather,
        (default 0.5).
    start : int, optional
        A value between 0 and the number of time slices indicating the start time of a subset,
        (default None).
    end : int, optional
        A value between 0 and the number of time slices indicating the end time of a subset,
        (default None)
    calc_ssim : bool, optional
        Whether or not to calculate the ssim (structural similarity index) between two plots
        (only applies to plot_type = 'spatial'), (default False)
    short_title: bool, optional
        If True, use a shortened title in the plot output (default False).
    axes_symmetric: bool, optional
        Whether or not to make the colorbar axes symmetric about zero (used in a spatial plot)
        (default False)
    legend_loc: str, optional
        The location to put the legend in a time-series plot in single-column format
        (plot_type = "time_series", vert_plot=True)
        (default "upper right")
    vert_plot: bool, optional
        If true, forces plots into a single column format and enlarges text.
        (default False)
    tex_format: bool, optional
        Whether to interpret all plot output strings as latex formatting (default False)
    legend_offset: 2-tuple, optional
        The x- and y- offset of the legend. Moves the corner of the legend specified by
        legend_loc to the specified location specified (where (0,0) is the bottom left corner
        of the plot and (1,1) is the top right corner). Only affects time-series, histogram,
        and periodogram plots.

    Returns
    =======
    out : None
    """

    mp = calcsPlot(
        ds,
        varname,
        calc,
        sets,
        group_by,
        scale,
        calc_type,
        plot_type,
        transform,
        subset,
        lat,
        lon,
        lev,
        color,
        quantile,
        calc_ssim=calc_ssim,
        legend_loc=legend_loc,
        axes_symmetric=axes_symmetric,
        short_title=short_title,
        vert_plot=vert_plot,
        tex_format=tex_format,
        legend_offset=legend_offset,
        weighted=weighted,
    )

    plt.rcParams.update(
        {
            'text.usetex': tex_format,
        }
    )

    mp.verify_plot_parameters()

    # Subset data (by var and collection)
    dss = []

    # update when new release of cf_xarray is released (won't need to do this - just trying to avoid
    # an uneeded arror message for now)
    if 'bounds' in ds['time'].attrs.keys():
        ds['time'].attrs.pop('bounds')

    # if varname == 'T':  # work around for cf_xarray (until new tag that
    #     # includes issue 130 updated to main on 1/27/21)
    #     ds.T.attrs['standard_name'] = 'tt'
    #     if 'collection' in ds[varname].dims:
    #         if sets is not None:
    #             for set in sets:
    #                 d = ds.cf['tt'].sel(collection=set)
    #                 d.coords["cell_area"] = ds.coords["cell_area"]
    #                 dss.append(d)
    #     else:
    #         d = ds.cf['tt']
    #         d.coords["cell_area"] = ds.coords["cell_area"]
    #         dss.append(ds.cf['tt'])
    #
    # else:

    if 'collection' in ds[varname].dims:
        if sets is not None:
            for set in sets:
                dss.append(ds[varname].sel(collection=set))
    else:
        dss.append(ds[varname])

    subsets = []
    if sets is not None:
        for i in range(len(sets)):
            subsets.append(lu.subset_data(dss[i], subset, lat, lon, lev, start, end))
            subsets[i].attrs = dss[i].attrs
            subsets[i].attrs['cell_measures'] = 'area: cell_area'

    # Acquire raw calc values
    datas = []
    if calc_type in ['calc_of_diff']:
        if subsets is not None:
            for i in range(1, len(subsets)):
                datas.append(subsets[0] - subsets[i])
                datas[i - 1].attrs = subsets[0].attrs
    else:
        if subsets is not None:
            for i in range(len(subsets)):
                datas.append(subsets[i])

    raw_calcs = []

    for d in datas:
        raw_calcs.append(mp.get_calcs(d))

    # get lat/lon coordinate names:
    if ds.data_type == 'pop':
        lon_coord_name = datas[0].cf[datas[0].cf.coordinates['longitude'][0]].dims[1]
        lat_coord_name = datas[0].cf[datas[0].cf.coordinates['latitude'][0]].dims[0]
    else:
        lat_coord_name = datas[0].cf[datas[0].cf.coordinates['latitude'][0]].dims[0]
        lon_coord_name = datas[0].cf[datas[0].cf.coordinates['longitude'][0]].dims[0]

    # Get calc names/values for plot title
    calc_names = []
    for i in range(len(datas)):
        if ds.variables.mapping.get('gw') is not None:
            calc_names.append(mp.get_calc_label(calc, datas[i], ds['gw'].values))
        else:
            calc_names.append(mp.get_calc_label(calc, datas[i]))
    # Get plot data and title
    if lat is not None and lon is not None:
        # is this a 1D of 2D lat/lon?
        dd = subsets[0].cf['latitude'].dims
        if len(dd) == 1:
            mp.title_lat = subsets[0][lat_coord_name].data[0]
            mp.title_lon = subsets[0][lon_coord_name].data[0] - 180
        else:  # 2
            # lon should be 0- 360
            mylat = subsets[0][lat_coord_name].data[0]
            mylon = subsets[0][lon_coord_name].data[0]
            if mylon < 0:
                mylon = mylon + 360
            mp.title_lat = mylat
            mp.title_lon = mylon
    else:
        mp.title_lat = lat
        mp.title_lon = lon

    plot_datas = []
    set_names = []

    if calc_type in ['diff', 'ratio']:
        for i in range(1, len(raw_calcs)):
            plot_datas.append(mp.get_plot_data(raw_calcs[0], raw_calcs[i]))
            set_names.append(tex_escape(f'{sets[0]} & {sets[i]}'))
    else:
        for i in range(len(raw_calcs)):
            plot_datas.append(mp.get_plot_data(raw_calcs[i]))
            if calc_type in ['calc_of_diff']:
                set_names.append(tex_escape(f'{sets[0]} & {sets[i+1]}'))
            else:
                set_names.append(f'{sets[i]}')

    plot_dataset = xr.concat(plot_datas, 'sets')
    plot_dataset = plot_dataset.assign_coords({'sets': set_names})

    titles = []

    if calc_type in ['ratio', 'diff']:
        for i in range(1, len(calc_names)):
            titles.append(mp.get_title(calc_names[i], f'{sets[0]} & {sets[i]}'))
    elif calc_type in ['calc_of_diff']:
        for i in range(len(calc_names)):
            titles.append(mp.get_title(calc_names[i], f'{sets[0]} & {sets[i+1]}'))
    else:
        for i in range(len(calc_names)):
            titles.append(mp.get_title(calc_names[i], sets[i]))

    # Call plot functions
    if plot_type == 'spatial':
        mp.spatial_plot(plot_dataset, titles)
    elif plot_type == 'time_series':
        mp.time_series_plot(plot_dataset, titles)
    elif plot_type == 'histogram':
        mp.hist_plot(plot_dataset, titles)
    elif plot_type == 'periodogram':
        mp.periodogram_plot(plot_dataset, titles)
