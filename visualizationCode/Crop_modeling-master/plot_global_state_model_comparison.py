import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

from plot_model_comparisons import load_prediction
from func_crop_model import prediction_result_by_state

## Calculate the model prediction performance at state level
def get_model_performance_state(model, yield_type='rainfed', prediction_type='forward',state_train=False):
    d = load_prediction(model, yield_type=yield_type, prediction_type=prediction_type,state=state_train)
    d_r2 = prediction_result_by_state(d, yield_type=yield_type)
    d_r2.loc[:,'rmse']=d_r2.loc[:,'rmse'] * 0.0628 # convert to t/ha
    return d_r2


# Get boxplot data for the model performance
def get_boxplot_data(df_r2_g,df_r2_s,var):
    temp1 = df_r2_g[var].to_frame(var)
    temp1['Model']='Global'
    
    temp2 = df_r2_s[var].to_frame(var)
    temp2['Model']='State'
    
    return pd.concat([temp1, temp2]).reset_index()

def make_plot(model='vpd_poly_evi_poly',yield_type='rainfed', prediction_type='leave_one_year_out'):
    # Load state model performance
    d_r2 = get_model_performance_state(model, yield_type=yield_type, prediction_type=prediction_type,state_train=True)
   # d = load_prediction(model, yield_type=yield_type, prediction_type=prediction_type,state=True)
   # d_r2 = prediction_result_by_state(d)
    
    # Load global model performance
    dg_r2 = get_model_performance_state(model, yield_type=yield_type, prediction_type=prediction_type,state_train=False)
    d2 = load_prediction(model, yield_type=yield_type,prediction_type=prediction_type)
    
    # Get mean county number of each state across years
    county_n = d2[d2.year>=2007].dropna().groupby(['year','State']).count().mean(level=1)['yield_rainfed'].apply(np.ceil)
    county_n.index = county_n.index.map(lambda x: x.title())

   # dg_r2 = prediction_result_by_state(d2)
    
    # 
    if ('evi' in model)|('lst' in model):
        year_begin = 2007
    else:
        year_begin = 2005
    
    d3_r2=get_boxplot_data(dg_r2.loc[(slice(None),slice(year_begin,2016)),:],d_r2.loc[(slice(None),slice(year_begin,2016)),:],'R2')
    d3_rmse=get_boxplot_data(dg_r2.loc[(slice(None),slice(year_begin,2016)),:],d_r2.loc[(slice(None),slice(year_begin,2016)),:],'rmse')
    
    d3_r2.loc[:,'State'] = d3_r2.State.apply(lambda x: x.title())
    d3_rmse.loc[:,'State'] = d3_rmse.State.apply(lambda x: x.title())
    
    # Begin plot
    fig, [ax1,ax2] = plt.subplots(2,1,figsize=(11,5.5))
    
    meanlineprops = dict(linestyle='--', color='k')
    
    sns.boxplot(x='State', y='R2',hue='Model', data=d3_r2,fliersize=0,ax=ax1,meanline=True,
                                                    showmeans=True,palette='Set3',meanprops=meanlineprops)
    sns.boxplot(x='State', y='rmse',hue='Model', data=d3_rmse,fliersize=0,ax=ax2,meanline=True,
                                                    showmeans=True,palette='Set3',meanprops=meanlineprops)
    
    ax1.set_ylabel('R2', fontsize=12)
    ax2.set_ylabel('RMSE (t/ha)', fontsize=12)
    
    ax1.text(0.01, 0.90, 'a', fontsize=12, transform=ax1.transAxes, fontweight='bold')
    ax2.text(0.01, 0.90, 'b', fontsize=12, transform=ax2.transAxes, fontweight='bold')
    
    
    # Create fake lines for legend
    mean_line = mlines.Line2D([], [], linestyle='--', color=ax1.artists[0].get_edgecolor()) 
    median_line = mlines.Line2D([], [], linestyle='-', color=ax1.artists[0].get_edgecolor())
    
    ax1.legend_.set_frame_on(False)
    # Rotate last row xticklabels
    # ax1.set_xticklabels(ax1.get_xticklabels(), rotation=10)
    # ax2.set_xticklabels(ax2.get_xticklabels(), rotation=10)
    
    ax2.legend([median_line,mean_line],['median','mean'],loc='upper right')

# add county number
    for i,s in enumerate(county_n.values):
        ax2.text(i, -0.5,'(%d)'%s, fontsize=10,ha='center')

    ax2.legend_.set_frame_on(False)
    
    ax2.set_ylim([0, 2.5]) # 40 bu *0.0628
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    
    plt.subplots_adjust(left=0.05,right=0.975,hspace=0.3,top=0.95)
    
    plt.savefig('../figure/figure_global_state_model_comparison_%s_%s_%s_test.pdf'%(model,yield_type,prediction_type))
   # plt.savefig('../figure/test.pdf')
    print('figure saved')

if __name__ == '__main__':
   # make_plot(model='vpd_poly',prediction_type='forward')
    make_plot()
