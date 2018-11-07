import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from func_crop_model import prediction_result_global
from plot_model_comparisons import load_prediction


# Constrain to seven states due to CDL availiability
state_2003=['ILLINOIS','INDIANA', 'IOWA','MISSOURI','NEBRASKA','NORTH DAKOTA','WISCONSIN']

# Best climate model
d0 = load_prediction('vpd_spline', yield_type='rainfed', prediction_type='leave_one_year_out')
d0_r2 = prediction_result_global(d0[d0.State.isin(state_2003)], yield_type='rainfed')
d0_r2.loc[:,'rmse']=d0_r2.loc[:,'rmse'] * 0.0628 # convert to t/ha

# Best climate + evi
d = load_prediction('vpd_spline_evi_poly', yield_type='rainfed', prediction_type='leave_one_year_out')
d1_r2 = prediction_result_global(d[d.State.isin(state_2003)], yield_type='rainfed')
d1_r2.loc[:,'rmse']=d1_r2.loc[:,'rmse'] * 0.0628 # convert to t/ha

print(d0_r2)
print(d1_r2)


# Make plot
#fig, [ax1,ax2] = plt.subplots(1,2,figsize=(10,5))
fig, [ax1,ax2] = plt.subplots(1,2,figsize=(10,3.75))

d0_r2['R2'].plot(style='-o',ax=ax1,color='C1')
d1_r2['R2'].plot(style='-o',ax=ax1,color='C2')
ax1.set_title('$R^2$',fontsize=12)

d0_r2['rmse'].plot(style='-o',ax=ax2,color='C1')
d1_r2['rmse'].plot(style='-o',ax=ax2,color='C2')
ax2.set_title('RMSE (t/ha)',fontsize=12)


#ax1.set_xticks(range(2003,2017))
#ax2.set_xticks(range(2003,2017))
#ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=12)
#ax2.set_xticklabels(ax2.get_xticklabels(), fontsize=12)


ax1.set_xlim([2002.5, 2016.5])
ax2.set_xlim([2002.5, 2016.5])

ax1.set_xlabel('Year',fontsize=12)
ax2.set_xlabel('Year',fontsize=12)

ax1.text(0.01, 0.95, 'a', fontsize=12, transform=ax1.transAxes, fontweight='bold')
ax2.text(0.01, 0.95, 'b', fontsize=12, transform=ax2.transAxes, fontweight='bold')

ax1.legend([u'Best Climate\u2013only','Best Climate + EVI'],loc='lower left')
ax1.legend_.set_frame_on(False)

plt.subplots_adjust(left=0.05,right=0.95,bottom=0.15, top=0.925)

#plt.savefig('../figure/figure_model_performance_interannual_variability.pdf')

print('figure saved')
