import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from func_crop_model import prediction_result_global

global yield_type_dict,area_type_dict
yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
area_type_dict = {'all': 'area', 'rainfed':'area_rainfed','irrigated':'area_irr'}


# Load the saved predicted results 
def load_prediction(model, yield_type='rainfed', prediction_type='forward',state=False, direct_fn=False):
    if state:
        df = pd.read_csv('../result/prediction_%s_%s_%s_state.csv'%(model,yield_type,prediction_type),
            dtype={'FIPS':str})
    else:    
        df = pd.read_csv('../result/prediction_%s_%s_%s.csv'%(model,yield_type,prediction_type),
                    dtype={'FIPS':str})
    if direct_fn:
        df = pd.read_csv('../result/' + direct_fn, dtype={'FIPS':str})

    return df[['year','FIPS','State','Predicted_'+yield_type_dict[yield_type],
           yield_type_dict[yield_type],area_type_dict[yield_type]]]


# Calculate the model prediction performance 
def get_model_performance(model, yield_type='rainfed', prediction_type='forward'):
    d = load_prediction(model, yield_type=yield_type, prediction_type=prediction_type)
    d_r2 = prediction_result_global(d, yield_type=yield_type)
    d_r2.loc[:,'rmse']=d_r2.loc[:,'rmse'] * 0.0628 # convert to t/ha
    return d_r2

def make_plot(yield_type='rainfed',prediction_type='forward'):
    # Climate variable selection
    #prediction_type = 'leave_one_year_out'
    
    d0_r2 = get_model_performance('Tgs_linear', yield_type=yield_type,prediction_type=prediction_type)
    d1_r2 = get_model_performance('tave_linear', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('vpd_linear', yield_type=yield_type,prediction_type=prediction_type)
    panel_1 = pd.concat([d0_r2,d1_r2,d2_r2], axis=0,keys=['$(T_{gs}$+$P_{gs})_{linear}$','$(T+P)_{linear}$','$(VPD+P)_{linear}$'])
    
    
    # Fitting functions 
    d0_r2 = get_model_performance('vpd_linear', yield_type=yield_type,prediction_type=prediction_type)
    d1_r2 = get_model_performance('vpd_poly', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('vpd_spline', yield_type=yield_type,prediction_type=prediction_type)
    panel_2 = pd.concat([d0_r2,d1_r2,d2_r2], axis=0,keys=['$(VPD+P)_{linear}$','$(VPD+P)_{poly}$','$(VPD+P)_{spline}$'])
    
    # Satellite data
    d0_r2 = get_model_performance('lstmax_spline_only', yield_type=yield_type,prediction_type=prediction_type)
    # d1_r2 = get_model_performance('lstmax_poly_only', yield_type='rainfed',prediction_type=prediction_type)
    d1_r2 = get_model_performance('evi_linear_only', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('evi_poly_only', yield_type=yield_type,prediction_type=prediction_type)
    d3_r2 = get_model_performance('lstmax_spline_evi_poly_only', yield_type=yield_type,prediction_type=prediction_type)
    panel_3 = pd.concat([d0_r2,d1_r2,d2_r2,d3_r2], axis=0,keys=['$LST_{spline}$','$EVI_{linear}$',
                                                                '$EVI_{poly}$','$LST_{spline}+EVI_{poly}$'])
    
    # Satellite data + Climate
    d0_r2 = get_model_performance('vpd_spline_evi', yield_type=yield_type,prediction_type=prediction_type)
    d1_r2 = get_model_performance('tave_spline_evi_poly', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('lstmax_spline_evi_poly', yield_type=yield_type,prediction_type=prediction_type)
    d3_r2 = get_model_performance('vpd_spline_evi_poly', yield_type=yield_type,prediction_type=prediction_type)
    panel_4 = pd.concat([d0_r2,d1_r2,d2_r2,d3_r2], axis=0,keys=['$(VPD+P)_{spline}+EVI_{linear}$',
                                                          '$(T+P)_{spline}+EVI_{poly}$',
                                                          '$(LST+P)_{spline}+EVI_{poly}$',
                                                         '$(VPD+P)_{spline}+EVI_{poly}$'])
    
    # Make plot
    fig, axes = plt.subplots(4,2, figsize=(13,10))
    lines = panel_1.loc[:,'R2'].unstack().T.boxplot(showfliers=False,ax=axes[0,0],grid=False,meanline=True,showmeans=True,widths=0.25)
    
    # Add legend
    lines = panel_1.loc[:,'rmse'].unstack().T.boxplot(showfliers=False,ax=axes[0,1],grid=False,meanline=True,
                                                    showmeans=True,widths=0.25,return_type='dict')
    
    axes[0,1].legend((lines['medians'][0],lines['means'][0]), ('median', 'mean'),loc='upper right',frameon=False)
    
    
    panel_2.loc[:,'R2'].unstack().T.boxplot(showfliers=False,ax=axes[1,0],grid=False,meanline=True,showmeans=True,widths=0.25)
    panel_2.loc[:,'rmse'].unstack().T.boxplot(showfliers=False,ax=axes[1,1],grid=False,meanline=True,showmeans=True,widths=0.25)
    
    panel_3.loc[:,'R2'].unstack().T.boxplot(showfliers=False,ax=axes[2,0],grid=False,meanline=True,showmeans=True,widths=0.25)
    panel_3.loc[:,'rmse'].unstack().T.boxplot(showfliers=False,ax=axes[2,1],grid=False,meanline=True,showmeans=True,widths=0.25)
    
    panel_4.loc[:,'R2'].unstack().T.boxplot(showfliers=False,ax=axes[3,0],grid=False,meanline=True,showmeans=True,widths=0.25)
    panel_4.loc[:,'rmse'].unstack().T.boxplot(showfliers=False,ax=axes[3,1],grid=False,meanline=True,showmeans=True,widths=0.25)
    
    # Rotate last row xticklabels
    axes[3,0].set_xticklabels(axes[3,0].get_xticklabels(), rotation=10,fontsize=12)
    axes[3,1].set_xticklabels(axes[3,1].get_xticklabels(), rotation=10,fontsize=12)
    
    # Change xlabel size: 
    for axx in axes.flatten()[:-1]:
        axx.set_xticklabels(axx.get_xticklabels(), fontsize=12)
    
    axes[0,0].set_title('R2',fontsize=14)
    axes[0,1].set_title('RMSE (t/ha)',fontsize=14)
    
    
    axes[0,1].text(1.1, 1.075, 'Climate variables', transform=axes[0,0].transAxes, fontsize=14, 
                   ha='center')
    axes[1,1].text(1.1, 1.075, 'Fitting functions', transform=axes[1,0].transAxes, fontsize=14, 
                   ha='center')
    axes[2,1].text(1.1, 1.075, 'Satellite variables', transform=axes[2,0].transAxes, fontsize=14, 
                   ha='center')
    axes[3,1].text(1.1, 1.075, 'Climate + Satellite variables', transform=axes[3,0].transAxes, fontsize=14, 
                   ha='center')
    
    for i,s in enumerate([chr(i) for i in range(ord('a'),ord('h')+1)]):
        axes.flatten()[i].text(0.01, 0.90, s, fontsize=14, transform=axes.flatten()[i].transAxes, fontweight='bold')
    
    
    plt.subplots_adjust(left=0.05,right=0.95,top=0.925, bottom=0.075, hspace=0.35)
    
    
    # for i in range(0,4):
    #     axes[i,0].set_ylim(0.45,0.9) 
    
    plt.savefig('../figure/figure_model_comparison_%s_%s.pdf'%(yield_type,prediction_type))
    print('figure saved')

if __name__ == '__main__':
    make_plot(yield_type='all',prediction_type='leave_one_year_out')
