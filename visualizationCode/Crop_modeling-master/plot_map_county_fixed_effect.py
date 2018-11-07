import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs

from plot_map_prediction_performance import norm_cmap,my_colormap
from func_crop_model import init_model,load_yield_data,define_model_structure_test,yield_trend


# First get the fixed effect from the trained model 

def get_fixed_effect(model_type = 'vpd_spline_evi_poly',yield_type = 'rainfed',y=2016,rerun=False):
    if rerun:
        df = load_yield_data()
    #     yield_type = 'rainfed'
    #     model_type = 'vpd_spline_evi_poly'

        model_txt = define_model_structure_test(model_type, yield_type=yield_type)

    #     y=2016
        corn_percent_min = 0

        train_data = df[(df['year']!=y)&(df['corn_percent']>corn_percent_min)]

        test_data = df[(df['year']==y)&(df['corn_percent']>corn_percent_min)]

        trend_results = yield_trend(df, yield_type=yield_type)

        # only predict county in train data
        con_fips = test_data['FIPS'].isin(train_data['FIPS'].unique())

        # If predict irrigated yield but there is no valid data in the training, 
        # E.g., Illinois after the state subset, set the predicted values to be nan

        m, df_predict = init_model(train_data, test_data[con_fips], model_txt, yield_type=yield_type)

        # Get model parameters and convert to data frame format
        c = m.params

        cc=c[c.index.str.contains('FIPS')].reset_index() #.to_frame()

        cc['FIPS']=cc['index'].apply(lambda x:x[10:15])

        cc.rename(columns={0:'fixed_effect'},inplace=True)
        cc[['FIPS','fixed_effect']].to_csv('../data/result/fixed_effect.csv',index=False)
        print('fixed effect saved to csv file')
    else:
        print('load the saved fixed effect from csv file')
        #TODO
        cc = pd.read_csv('../data/result/fixed_effect.csv',dtype={'FIPS':str})
    return cc
    

def plot_map(df,ax):
    ax.set_extent([-105, -80, 35, 49], ccrs.Geodetic())

    fips_list = df.index.tolist()

    # Plot county value    
    for record, county in zip(county_shapes.records(), county_shapes.geometries()):
        fips = record.attributes['FIPS']
        if fips in fips_list:
            facecolor = df.loc[fips,'color']
            ax.add_geometries([county], ccrs.PlateCarree(),
              facecolor=facecolor, edgecolor='white', linewidth=0)
    #     else:
    #         facecolor = 'grey'


    # Plot state boundary    
    for state in state_shapes.geometries():
        facecolor = 'None'
        ax.add_geometries([state], ccrs.PlateCarree(),
                          facecolor=facecolor, edgecolor='black',linewidth=0.6)



# Plot map of the fixed effect

# Load shapefile
global county_shapes, state_shapes
shapefile='./counties_contiguous/counties_contiguous.shp'
county_shapes = shpreader.Reader(shapefile)
shapefile='./states_contiguous/states_contiguous.shp'
state_shapes = shpreader.Reader(shapefile)

cc = get_fixed_effect()

mycmap = my_colormap(name='rainbow', customize=False)

vmin=-100
vmax=30

cmap_r, norm_r = norm_cmap(cmap=mycmap,
                       vmin=vmin, vmax=vmax)

cc['color'] = [cmap_r.to_rgba(value) for value in cc['fixed_effect'].values]

subplot_kw = dict(projection=ccrs.LambertConformal())

fig, ax1 = plt.subplots(1,1,subplot_kw=subplot_kw)

plot_map(cc.set_index('FIPS'),ax1)

ax1.set_title('County fixed effect in the "Best Climate + EVI Model"')

#colorbar
cax = fig.add_axes([0.85, 0.2, 0.015, 0.6])
cb1 = mpl.colorbar.ColorbarBase(ax=cax, cmap=cmap_r.cmap,
                                    norm=norm_r,
                                    orientation='vertical')
cb1.ax.set_ylabel('$r$',rotation=270,labelpad=10)

cb1.ax.set_ylabel('bu/acres',rotation=270,labelpad=10)


plt.subplots_adjust(top=0.95,bottom=0.05,left=0.05,right=0.8)

plt.savefig('./figure/figure_map_county_fixed_effect.pdf')

print('map saved')
#plt.show()
