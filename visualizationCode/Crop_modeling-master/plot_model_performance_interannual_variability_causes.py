import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from func_crop_model import prediction_result_global, load_yield_data
from plot_model_comparisons import load_prediction

# Plot scatter values
def plot_name(df, x_txt, y_txt, ax, fontsize=8):
    for i in df.index.values:
        ax.text(df.loc[i,x_txt],df.loc[i,y_txt], i, fontsize=fontsize)   
        
# Plot fitting         
def plot_fitting(df,x_txt, y_txt, ax, order=2):
    p = np.poly1d(np.polyfit(df[x_txt], df[y_txt], order))
    xp = np.linspace(df[x_txt].min(), df[x_txt].max(), 100)
    ax.plot(xp, p(xp), '-')

def make_plot(model='vpd_spline_evi_poly'):
    data_12 = load_yield_data()
    data_12.loc[:,'yield_rainfed'] = data_12.loc[:,'yield_rainfed'] * 0.0628

    # Constrain to seven states due to CDL availiability
    state_2003=['ILLINOIS','INDIANA', 'IOWA','MISSOURI','NEBRASKA','NORTH DAKOTA','WISCONSIN']

#    state_2003 = ['ILLINOIS', 'INDIANA', 'IOWA', 'KANSAS', 'MICHIGAN', 'MINNESOTA',
#       'MISSOURI', 'NEBRASKA', 'NORTH DAKOTA', 'OHIO', 'SOUTH DAKOTA', 'WISCONSIN']
    
    # Load best climate + evi
    d = load_prediction(model, yield_type='rainfed', prediction_type='leave_one_year_out')
    d1_r2 = prediction_result_global(d[d.State.isin(state_2003)], yield_type='rainfed')
    d1_r2.loc[:,'rmse']=d1_r2.loc[:,'rmse'] * 0.0628 # convert to t/ha

    d_pre = d1_r2.join(data_12[data_12.State.isin(state_2003)].groupby('year').mean()['precip_gs_z'].to_frame())
    d_yieldstd = d1_r2.join(data_12[data_12.State.isin(state_2003)].groupby('year').std()['yield_rainfed'].to_frame())


    fig, axes = plt.subplots(2,2,figsize=(10,7.5))
    d_pre.plot.scatter(x='precip_gs_z',y='R2',ax=axes[0,0])
    
    plot_name(d_pre, 'precip_gs_z','R2',axes[0,0])
    plot_fitting(d_pre, 'precip_gs_z','R2',axes[0,0],order=2)
    
    d_pre.plot.scatter(x='precip_gs_z',y='rmse',ax=axes[0,1])
    plot_name(d_pre, 'precip_gs_z','rmse',axes[0,1])
    plot_fitting(d_pre, 'precip_gs_z','rmse',axes[0,1],order=2)
    
    axes[0,0].set_title('$R^2$')
    axes[0,1].set_title('RMSE (t/ha)')
    
    axes[0,0].set_xlabel('Precipitation standard anomaly',fontsize=12)
    axes[0,1].set_xlabel('Precipitation standard anomaly',fontsize=12)
    
    axes[0,0].set_ylabel('')
    axes[0,1].set_ylabel('')
    
    
    d_yieldstd.plot.scatter(x='yield_rainfed',y='R2',ax=axes[1,0])
    plot_name(d_yieldstd, 'yield_rainfed','R2',axes[1,0])
    plot_fitting(d_yieldstd,'yield_rainfed','R2',axes[1,0],order=1)
    
    
    d_yieldstd.plot.scatter(x='yield_rainfed',y='rmse',ax=axes[1,1])
    plot_name(d_yieldstd, 'yield_rainfed','rmse',axes[1,1])
    plot_fitting(d_yieldstd,'yield_rainfed','rmse',axes[1,1],order=1)
    
    axes[1,0].set_xlabel('Spatial yield variability (t/ha)',fontsize=12)
    axes[1,1].set_xlabel('Spatial yield variability (t/ha)',fontsize=12)
    
    axes[1,0].set_ylabel('')
    axes[1,1].set_ylabel('')
    
    # Add panel label 
    for i,s in enumerate([chr(i) for i in range(ord('a'),ord('d')+1)]):
        axes.flatten()[i].text(0.01, 0.95, s, fontsize=12, transform=axes.flatten()[i].transAxes, fontweight='bold')
    
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.1,hspace=0.25)
    
   # plt.savefig('../figure/figure_model_performance_interannual_variability_causes_12states.pdf')
    plt.savefig('../figure/figure_model_performance_interannual_variability_causes_%s.pdf'%model)
   # plt.savefig('../figure/test.pdf')
    print('figure saved')

if __name__ == "__main__":
   # make_plot(model='vpd_spline')
    make_plot()
