import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import matplotlib as mpl
from matplotlib.colors import Normalize
import sys

sys.path.insert(0, '../../predictionCode')
from print_model_comparisons import load_prediction
import os

homePath = os.getcwd()

GET_AVERAGE_STATS_BIAS = -1
GET_AVERAGE_PREDICTED_YIELD = -2
GET_AVERAGE_TRUE_YIELD = -3


def norm_cmap(cmap, vmin=None, vmax=None):
    """
    Normalize and set colormap
    
    Parameters
    ----------
    values : Series or array to be normalized
    cmap : matplotlib Colormap
    normalize : matplotlib.colors.Normalize
    cm : matplotlib.cm
    vmin : Minimum value of colormap. If None, uses min(values).
    vmax : Maximum value of colormap. If None, uses max(values).
    
    Returns
    -------
    n_cmap : mapping of normalized values to colormap (cmap)
    
    """
    #     mn = vmin or min(values)
    #     mx = vmax or max(values)
    #     norm = Normalize(vmin=mn, vmax=mx)
    norm = Normalize(vmin=vmin, vmax=vmax)
    n_cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return n_cmap, norm


# My own colormap or matplotlib default colormap
def my_colormap(name='RdBu', customize=False):
    # Combine from different colormaps
    if customize:
        red = plt.cm.RdBu(np.linspace(0., 0.5, 128))
        green = plt.cm.PRGn(np.linspace(0.5, 1, 128))
        blue = plt.cm.RdBu(np.linspace(1, 0.5, 128))

        if name == 'BuGr':
            # combine them and build a new colormap
            # https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps
            colors = np.vstack((blue, green))
        if name == 'RdGr':
            colors = np.vstack((red, green))

        cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
    else:
        cmap = plt.get_cmap(name)
    return cmap


def plot_map(df, ax, year=GET_AVERAGE_STATS_BIAS):
    ax.set_extent([-105, -80, 35, 49], ccrs.Geodetic())

    # TODO
    # I changed df.n >= 5 to df.n >= 1 here
    if year == GET_AVERAGE_STATS_BIAS:
        fips_list = df.loc[df.n >= 5].index.tolist()
    else:
        fips_list = df.loc[df.n >= 1].index.tolist()

    # Plot county value    
    for record, county in zip(county_shapes.records(), county_shapes.geometries()):
        fips = record.attributes['FIPS']
        if fips in fips_list:
            facecolor = df.loc[fips, 'color']
            ax.add_geometries([county], ccrs.PlateCarree(),
                              facecolor=facecolor, edgecolor='white', linewidth=0)
    #     else:
    #         facecolor = 'grey'

    # Plot state boundary
    for state in state_shapes.geometries():
        facecolor = 'None'
        ax.add_geometries([state], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor='black', linewidth=0.6)


# Calculate RMSE
def cal_rmse(d, col_p, col_true):
    return d[col_p] - d[col_true]


def cal_rmse_second(d, col_p, col_true):
    return d[col_p]


def cal_rmse_third(d, col_p, col_true):
    return d[col_true]


# def cal_rmse(d,col_p,col_true):
# return ((d[col_p] - d[col_true])**2/d.shape[0])**0.5

# Also add RMSE
def get_county_performance(model='vpd_spline_evi_poly', yield_type='rainfed', prediction_type='leave_one_year_out',
                           year=GET_AVERAGE_STATS_BIAS):
    # Load data
    yield_type_dict = {'all': 'yield', 'rainfed': 'yield_rainfed', 'irrigated': 'yield_irr'}

    os.chdir("../../dataFiles/newR_Prediction_CSVs")
    df = load_prediction(model, yield_type=yield_type, prediction_type=prediction_type)
    df = df.loc[~pd.isnull(df['yield_rainfed']), :]
    os.chdir(homePath)
    if year != GET_AVERAGE_STATS_BIAS:
        df = df[df['year'] == y]

    county_r = df.groupby('FIPS')[['Predicted_' + yield_type_dict[yield_type],
                                   yield_type_dict[yield_type]]].corr().loc[(slice(None), yield_type_dict[yield_type]),
                                                                            'Predicted_' + yield_type_dict[yield_type]]
    county_r.index = county_r.index.droplevel(1)

    county_n = df.dropna().groupby('FIPS')[['Predicted_' + yield_type_dict[yield_type],
                                            yield_type_dict[yield_type]]].count()[yield_type_dict[yield_type]]

    county_rn = pd.concat([county_r.rename('r'), county_n.rename('n')], axis=1)
    county_rn['St'] = county_rn.index.to_series().apply(lambda x: x[0:2])

    if year == GET_AVERAGE_STATS_BIAS:
        county_rmsen = df.dropna().set_index('FIPS').groupby(level=0).apply(cal_rmse, 'Predicted_yield_rainfed',
                                                                            'yield_rainfed').mean(level=0)
    # The following returns the average of 'Predicted_yield_rainfed'
    elif year == GET_AVERAGE_PREDICTED_YIELD:
        county_rmsen = df.dropna().set_index('FIPS').groupby(level=0).apply(cal_rmse_second, 'Predicted_yield_rainfed',
                                                                            'yield_rainfed').mean(level=0)
    # The following returns the average of 'yield_rainfed'
    elif year == GET_AVERAGE_TRUE_YIELD:
        county_rmsen = df.dropna().set_index('FIPS').groupby(level=0).apply(cal_rmse_third, 'Predicted_yield_rainfed',
                                                                            'yield_rainfed').mean(level=0)
    else:
        county_rmsen = df.dropna().set_index('FIPS').groupby(level=0).apply(cal_rmse, 'Predicted_yield_rainfed',
                                                                            'yield_rainfed').median(level=0)
    county_rn = county_rn.join(county_rmsen.rename('rmse'))
    county_rn.loc[:, 'rmse'] = county_rn.loc[:, 'rmse']
    # county_rn.loc[:,'rmse'] = county_rn.loc[:,'rmse'] * 0.0628

    return county_rn


def make_plot(yield_type='rainfed', prediction_type='leave_one_year_out', year=GET_AVERAGE_STATS_BIAS):
    # Load shapefile
    global county_shapes, state_shapes
    shapefile = './counties_contiguous/counties_contiguous.shp'
    county_shapes = shpreader.Reader(shapefile)
    shapefile = './states_contiguous/states_contiguous.shp'
    state_shapes = shpreader.Reader(shapefile)

    mycmap = my_colormap(name='rainbow', customize=False)

    if year == GET_AVERAGE_PREDICTED_YIELD:
        county_r = get_county_performance(model='vpd_spline_evi_poly', yield_type=yield_type,
                                          prediction_type=prediction_type, year=GET_AVERAGE_TRUE_YIELD)
        county_rmse = get_county_performance(model='vpd_spline_evi_poly', yield_type=yield_type,
                                             prediction_type=prediction_type, year=GET_AVERAGE_PREDICTED_YIELD)

    else:
        county_r = get_county_performance(model='vpd_spline_evi_poly', yield_type=yield_type,
                                          prediction_type=prediction_type, year=year)
        county_rmse = county_r.copy()
    #    county_rmse = get_county_performance(model='vpd_spline_evi_poly',yield_type=yield_type, prediction_type=prediction_type)

    if year == GET_AVERAGE_PREDICTED_YIELD:
        cmap_r, norm_r = norm_cmap(cmap=mycmap,
                                   vmin=25, vmax=72)
    else:
        cmap_r, norm_r = norm_cmap(cmap=mycmap,
                                   vmin=0.3, vmax=1)
    if year == GET_AVERAGE_STATS_BIAS:
        cmap_rmse, norm_rmse = norm_cmap(cmap=mycmap, vmin=-3, vmax=3)
        # cmap_rmse, norm_rmse = norm_cmap(cmap=mycmap,
        # vmin=county_rmse['rmse'].min(), vmax=county_rmse['rmse'].max())
        # 0.2
    elif year == GET_AVERAGE_PREDICTED_YIELD:
        cmap_rmse, norm_rmse = norm_cmap(cmap=mycmap, vmin=25, vmax=72)
    else:
        cmap_rmse, norm_rmse = norm_cmap(cmap=mycmap, vmin=-11, vmax=16)
        # cmap_rmse, norm_rmse = norm_cmap(cmap=mycmap,
        # vmin=county_rmse['rmse'].min(), vmax=county_rmse['rmse'].max())
        # 0.5

    if year == GET_AVERAGE_PREDICTED_YIELD:
        county_r['color'] = [cmap_r.to_rgba(value) for value in county_r['rmse'].values]
    else:
        county_r['color'] = [cmap_r.to_rgba(value) for value in county_r['r'].values]

    county_rmse['color'] = [cmap_rmse.to_rgba(value) for value in county_rmse['rmse'].values]

    print((year, np.percentile(county_rmse['rmse'], 2), np.percentile(county_rmse['rmse'], 98)))

    subplot_kw = dict(projection=ccrs.LambertConformal())

    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(12.8, 4.8), subplot_kw=subplot_kw)

    plot_map(county_r, ax1, year)
    plot_map(county_rmse, ax2, year)

    if year == GET_AVERAGE_STATS_BIAS:
        ax1.set_title('($r^2$) from 2003 to 2016')
        ax2.set_title('Average (RMSE Bias: Predicted - Actual) over 2003 to 2016')

    elif year == GET_AVERAGE_PREDICTED_YIELD:
        ax1.set_title('Average Predicted over 2003 to 2016')
        ax2.set_title('Average Actual Yield over 2003 to 2016')

    else:
        ax1.set_title('($r$) in %s' % year)
        ax2.set_title('(RMSE Bias: Predicted - Actual) in %s' % year)

    if year == GET_AVERAGE_STATS_BIAS or year == GET_AVERAGE_PREDICTED_YIELD:
        ax1.text(-0.05, 1.025, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
        ax2.text(-0.05, 1.025, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    if year == GET_AVERAGE_STATS_BIAS or year == GET_AVERAGE_PREDICTED_YIELD:
        ax2.text(-0.05, 1.025, 'a', fontsize=14, transform=ax2.transAxes, fontweight='bold')

    # colorbar
    # cax = fig.add_axes([0.925, 0.2, 0.015, 0.6])
    if year == GET_AVERAGE_STATS_BIAS:
        cax = fig.add_axes([0.45, 0.2, 0.01, 0.6])
        cb1 = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap_r.cmap,
                                        norm=norm_r,
                                        orientation='vertical')
        cb1.ax.set_ylabel('$r$', rotation=270, labelpad=10)
    if year == GET_AVERAGE_PREDICTED_YIELD:
        cax = fig.add_axes([0.45, 0.2, 0.01, 0.6])
        cb1 = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap_r.cmap,
                                        norm=norm_r,
                                        orientation='vertical')
        cb1.ax.set_ylabel('$Average Actual Yield$', rotation=270, labelpad=10)

    cax2 = fig.add_axes([0.915, 0.2, 0.01, 0.6])
    cb2 = mpl.colorbar.ColorbarBase(ax=cax2, cmap=cmap_rmse.cmap,
                                    norm=norm_rmse,
                                    orientation='vertical')
    cb2.ax.set_ylabel('Yield Bias (bu/ac)', rotation=270, labelpad=15)

    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, hspace=0.25)

    os.chdir("../figures/mapPlots/")
    if year == GET_AVERAGE_STATS_BIAS:
        plt.savefig('figure_map_prediction_performance_%s_%s_all.png' % (yield_type, prediction_type))
    elif year == GET_AVERAGE_PREDICTED_YIELD or year == GET_AVERAGE_TRUE_YIELD:
        # f = plt.gcf()
        # f.delaxes(ax1)
        finalString = ""
        if (year == GET_AVERAGE_PREDICTED_YIELD):
            finalString = "Predicted_Actual_yield"

        plt.savefig('figure_map_prediction_performance_%s_%s_%s.png' % (yield_type, prediction_type, finalString))
    else:
        f = plt.gcf()
        f.delaxes(ax1)
        plt.savefig('figure_map_prediction_performance_%s_%s_%s.png' % (yield_type, prediction_type, str(year)))
    # plt.savefig('../figures/test.pdf')
    os.chdir(homePath)
    print('figure map saved')


if __name__ == '__main__':
    for y in range(2003, 2017, 1):
        make_plot(year=y)
    make_plot()
    make_plot(year=GET_AVERAGE_PREDICTED_YIELD)
