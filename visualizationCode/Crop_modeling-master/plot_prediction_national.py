import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_model_comparisons import load_prediction

# run under my_geo enviroment, otherwise the color changes, due to matploblib version?
global yield_type_dict
yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
area_type_dict = {'all': 'area', 'rainfed':'area_rainfed','irrigated':'area_irr'}

def get_national_prediction(model,yield_type='rainfed',prediction_type='forward',area_weight=False, 
                            direct_fn=False):
    df = load_prediction(model, yield_type=yield_type, prediction_type=prediction_type, 
                         direct_fn=direct_fn)
    df = df.dropna()
    state_2003=['ILLINOIS','INDIANA', 'IOWA','MISSOURI','NEBRASKA','NORTH DAKOTA','WISCONSIN']

    # Only 7 states before 2007
    con = df.State.isin(state_2003)&(df.year<2007)
  #  con = df.State.isin(state_2003)
    # all states after 2007
    con = con | (df.year>=2007)

    if area_weight:
        df['Predicted_'+yield_type_dict[yield_type]+'_area']=(df['Predicted_'+yield_type_dict[yield_type]]
                                                             *df[area_type_dict[yield_type]])
        df[yield_type_dict[yield_type]+'_area']=(df[yield_type_dict[yield_type]]
                                                *df[area_type_dict[yield_type]])   
        
        yield_predicted=df[con].groupby('year').sum()['Predicted_' + yield_type_dict[yield_type] + '_area']/ \
            df[con].groupby('year').sum()[area_type_dict[yield_type]]
        yield_actual=df[con].groupby('year').sum()[yield_type_dict[yield_type] + '_area']/ \
            df[con].groupby('year').sum()[area_type_dict[yield_type]]
    else:
        yield_predicted = df[con].groupby('year').mean()['Predicted_'+yield_type_dict[yield_type]]
        yield_actual = df[con].groupby('year').mean()[yield_type_dict[yield_type]]
    return yield_predicted,yield_actual

#def get_national_prediction(model,yield_type='rainfed',prediction_type='forward'):
#    df = load_prediction(model, yield_type=yield_type, prediction_type=prediction_type)
##     state_2003=['ILLINOIS','INDIANA', 'IOWA','MISSOURI','NEBRASKA','NORTH DAKOTA','WISCONSIN']
##     con = df.State.isin(state_2003)
#    con = df.year>=2005
#    yield_predicted = df[con].groupby('year').mean()['Predicted_'+yield_type_dict[yield_type]]
#    yield_actual = df[con].groupby('year').mean()[yield_type_dict[yield_type]]
#    return yield_predicted,yield_actual

yield_type = 'rainfed'
#yield_type = 'all'
#prediction_type = 'forward'
prediction_type = 'leave_one_year_out'
#fn = 'prediction_vpd_spline_evi_poly_rainfed_forward_cornpercent.csv'

fn = False

d1,d0 = get_national_prediction('vpd_spline_evi_poly', yield_type=yield_type, prediction_type=prediction_type,
                                 direct_fn=fn)

d2 = get_national_prediction('vpd_spline', yield_type=yield_type, prediction_type=prediction_type)[0]
d_final = pd.concat([d0.rename('Actual'),d1.rename('M1'),d2.rename('M2')], axis=1)
d_final = d_final * 0.0628


d1w,d0w = get_national_prediction('vpd_spline_evi_poly', yield_type=yield_type, prediction_type=prediction_type,
                                  area_weight=True, direct_fn=fn)
d2w = get_national_prediction('vpd_spline', yield_type=yield_type, prediction_type=prediction_type,area_weight=True)[0]
d_final2 = pd.concat([d0w.rename('Actual'),d1w.rename('M1'),d2w.rename('M2')], axis=1)
d_final2 = d_final2 * 0.0628

if prediction_type == 'leave_one_year_out':
    d_final2 = d_final2.loc[2005:]
    d_final = d_final.loc[2005:]

rmse = ((d_final.loc[:,'M1':'M2'].subtract(d_final.loc[:,'Actual'],axis=0)**2).sum()/d_final.shape[0])**0.5
rmse2 = ((d_final2.loc[:,'M1':'M2'].subtract(d_final2.loc[:,'Actual'],axis=0)**2).sum()/d_final2.shape[0])**0.5



# Make plot

fig,[ax1,ax2]=plt.subplots(1,2, figsize=(10,3.75))

d_final['Actual'].plot(style='-p',lw=2,ax=ax1)
d_final['M2'].plot(style='--o',lw=2,ax=ax1)
d_final['M1'].plot(style='--o',lw=2,ax=ax1)

#ax1.set_xticks(range(2005,2017))

ax1.set_ylabel('Yield (t/ha)',fontsize=12)
ax1.set_xlabel('Year',fontsize=12)
ax1.set_xlim([2004.2, 2016.5])
ax1.set_title('County average with equal weights')

ax1.legend(['Actual','Best climate','Best climate+EVI'],loc='lower left')
ax1.legend_.set_frame_on(False)

ax1.text(0.75,0.15,'RMSE:%.3f'%rmse['M1'],transform=ax1.transAxes,color=ax1.get_lines()[2].get_color())
ax1.text(0.75,0.10,'RMSE:%.3f'%rmse['M2'],transform=ax1.transAxes,color=ax1.get_lines()[1].get_color())

d_final2['Actual'].plot(style='-p',lw=2,ax=ax2)
d_final2['M2'].plot(style='--o',lw=2,ax=ax2)
d_final2['M1'].plot(style='--o',lw=2,ax=ax2)
ax2.set_xlim([2004.2, 2016.5])
ax2.set_xlabel('Year',fontsize=12)
ax2.set_ylabel('Yield (t/ha)',fontsize=12)
ax2.set_title('County weighted average by harvest area')
#ax2.set_ylim(ax1.get_ylim())

ax2.text(0.75,0.15,'RMSE:%.3f'%rmse2['M1'],transform=ax2.transAxes, color=ax2.get_lines()[2].get_color())
ax2.text(0.75,0.10,'RMSE:%.3f'%rmse2['M2'],transform=ax2.transAxes, color=ax2.get_lines()[1].get_color())

ax1.text(0.01, 0.95, 'a', fontsize=12, transform=ax1.transAxes, fontweight='bold')
ax2.text(0.01, 0.95, 'b', fontsize=12, transform=ax2.transAxes, fontweight='bold')

plt.subplots_adjust(left=0.05,right=0.95,bottom=0.15, top=0.925)

plt.savefig('../figure/figure_prediction_national_%s_%s.png'%(yield_type,prediction_type))
print('figure for %s %s saved'%(yield_type,prediction_type))
