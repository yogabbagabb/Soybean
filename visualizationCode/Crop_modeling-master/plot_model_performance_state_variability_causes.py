import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from func_crop_model import load_yield_data
from plot_model_performance_interannual_variability_causes import plot_name, plot_fitting
from plot_global_state_model_comparison import get_model_performance_state

#model = 'vpd_spline_evi_poly'
model = 'vpd_poly_evi_poly'
d_r2 = get_model_performance_state(model, yield_type='rainfed', prediction_type='leave_one_year_out')

data_12 = load_yield_data()
data_12.loc[:,'yield_rainfed'] = data_12.loc[:,'yield_rainfed'] * 0.0628 # convert to t/ha

# Change state name index 
d_yieldvar_t = d_r2.median(level=0).join(data_12.loc[data_12.year>=2003,:].groupby(['year','State']).mean()['yield_rainfed'].std(level=1).to_frame()).copy()
d_yieldvar_t.index = d_yieldvar_t.index.map(lambda x: x.title())

d_yieldvar_s = d_r2.median(level=0).join(data_12.loc[data_12.year>=2003,:].groupby(['year','State']).std()['yield_rainfed'].mean(level=1).to_frame()).copy()
d_yieldvar_s.index = d_yieldvar_s.index.map(lambda x: x.title())


fig, axes = plt.subplots(2,2,figsize=(10,7.5))
d_yieldvar_s.plot.scatter(x='yield_rainfed',y='R2',ax=axes[0,0])
plot_name(d_yieldvar_s, 'yield_rainfed','R2',axes[0,0],fontsize=10)
plot_fitting(d_yieldvar_s, 'yield_rainfed','R2',axes[0,0],order=1)

d_yieldvar_s.plot.scatter(x='yield_rainfed',y='rmse',ax=axes[0,1])
plot_name(d_yieldvar_s, 'yield_rainfed','rmse',axes[0,1],fontsize=10)
plot_fitting(d_yieldvar_s, 'yield_rainfed','rmse',axes[0,1],order=1)

d_yieldvar_t.plot.scatter(x='yield_rainfed',y='R2',ax=axes[1,0])
plot_name(d_yieldvar_t, 'yield_rainfed','R2',axes[1,0],fontsize=10)
plot_fitting(d_yieldvar_t, 'yield_rainfed','R2',axes[1,0],order=1)

d_yieldvar_t.plot.scatter(x='yield_rainfed',y='rmse',ax=axes[1,1])
plot_name(d_yieldvar_t, 'yield_rainfed','rmse',axes[1,1],fontsize=10)
plot_fitting(d_yieldvar_t, 'yield_rainfed','rmse',axes[1,1],order=1)

axes[0,0].set_title('R2',fontsize=12)
axes[0,1].set_title('RMSE (t/ha)',fontsize=12)


axes[0,0].set_xlabel('State spatial yield variability (t/ha)',fontsize=12)
axes[0,1].set_xlabel('State spatial yield variability (t/ha)',fontsize=12)

axes[0,0].set_ylabel('')
axes[0,1].set_ylabel('')

axes[1,0].set_xlabel('State temporal yield variability (t/ha)',fontsize=12)
axes[1,1].set_xlabel('State temporal yield variability (t/ha)',fontsize=12)

axes[1,0].set_ylabel('')
axes[1,1].set_ylabel('')

for i,s in enumerate([chr(i) for i in range(ord('a'),ord('d')+1)]):
    axes.flatten()[i].text(0.01, 0.925, s, fontsize=14, transform=axes.flatten()[i].transAxes, fontweight='bold')

plt.subplots_adjust(left=0.05,right=0.925,top=0.9,wspace=0.25,hspace=0.25)

plt.savefig('../figure/figure_model_performance_state_variability_causes_%s.pdf'%model)

print('figure saved')
