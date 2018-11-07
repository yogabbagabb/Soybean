import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from func_crop_model import yield_trend

lowess = sm.nonparametric.lowess

# Load data
data = pd.read_csv('../../dataFiles/soybean_model_data_2017.csv',dtype={'FIPS':str})

# Add logical filter to the yield Data
area_con = data['area'].notnull()
data = data[area_con]

# Add Rainfed yield
# rainfed_con: counties without irrigation, the yield is rainfed
rainfed_con = ~data['FIPS'].isin(data.loc[data['yield_irr'].notnull(),'FIPS'].unique())
data['yield_rainfed'] = data['yield_noirr']
data['area_rainfed'] = data['area_noirr']

# For counties with irrigation, only the rainfed yield is added to irrigated yield
data.loc[rainfed_con, 'yield_rainfed'] = data.loc[rainfed_con, 'yield']
data.loc[rainfed_con, 'area_rainfed'] = data.loc[rainfed_con, 'area']



data_12 = data[data['State'].isin(data.loc[data['evi6'].notnull(),'State'].unique())]

# Detrend yield
trend_rainfed = yield_trend(data_12, yield_type='rainfed')

data_12.loc[:,'yield_rainfed_ana'] = (data_12['yield_rainfed'] - trend_rainfed.predict(data_12[['year','yield_rainfed']]))      

# Convert to t/ha
data_12.loc[:,'yield_rainfed_ana'] = data_12.loc[:,'yield_rainfed_ana'] * 0.0628


# Plot figure
x_var = [['vpdave5','vpdave6','vpdave7','vpdave8','vpdave9'],['tave5','tave6','tave7','tave8','tave9'], 
         ['precip5','precip6','precip7','precip8','precip9'],
          ['evi5','evi6','evi7','evi8','evi9'],['lstmax5','lstmax6', 'lstmax7','lstmax8','lstmax9']]

y_var = 'yield_rainfed_ana'

con = data_12['year']>1980

fig, axes = plt.subplots(5,5, figsize=(12,6))

for row in range(0,5):
    for col in range(0,5):
        Z = lowess(data_12.loc[con,y_var],
                   data_12.loc[con, x_var[col][row]],
                   frac=0.3,it=3)
        
        axes[row,col].plot(Z[:,0], Z[:,1], 'r-', lw=4)
        axes[row,col].scatter(data_12.loc[con, x_var[col][row]], data_12.loc[con,y_var],s=1, 
                              color='grey',rasterized=True)
        if col !=0:
            axes[row,col].set_yticklabels('')

axes[0,0].set_title('VPD (hPa)')        
axes[0,1].set_title(u'Air temperature (\xb0C)')        
axes[0,2].set_title('Precipitation (mm)')
axes[0,3].set_title('EVI')        
axes[0,4].set_title(u'maxLST (\xb0C)')        

        
axes[0,-1].text(1.05,0.5,'May',transform=axes[0,-1].transAxes, fontsize=12)
axes[1,-1].text(1.05,0.5,'June',transform=axes[1,-1].transAxes, fontsize=12)        
axes[2,-1].text(1.05,0.5,'July',transform=axes[2,-1].transAxes, fontsize=12)        
axes[3,-1].text(1.05,0.5,'August',transform=axes[3,-1].transAxes, fontsize=12)        
axes[4,-1].text(1.05,0.5,'September',transform=axes[4,-1].transAxes, fontsize=12)        


axes[1,0].text(-0.4,0.7,'Rainfed Yield Anomaly (t/ha)',transform=axes[1,0].transAxes, 
               fontsize=10,rotation=90)

# Add knots

# VPD
axes[1,0].text(1,0.925,'knots: 8.5,10.5,12.5,14.96',transform=axes[1,0].transAxes, 
               fontsize=8,ha='right')
axes[2,0].text(1,0.925,'knots: 8, 10.5',transform=axes[2,0].transAxes, 
               fontsize=8,ha='right')
axes[3,0].text(1,0.925,'knots: 8.06, 16.5521',transform=axes[3,0].transAxes, 
               fontsize=8,ha='right')
# Tave
axes[1,1].text(1,0.925,'knots: 21.014,23.182',transform=axes[1,1].transAxes, 
               fontsize=8,ha='right')
axes[2,1].text(1,0.925,'knots: 22.5373,25.4477',transform=axes[2,1].transAxes, 
               fontsize=8,ha='right')
axes[3,1].text(1,0.925,'knots: 21.55,24.815',transform=axes[3,1].transAxes, 
               fontsize=8,ha='right')

# Precip
axes[1,2].text(1,0.925,'knots: 92.6, 209.06',transform=axes[1,2].transAxes, 
               fontsize=8,ha='right')
axes[2,2].text(1,0.925,'knots: 56.191, 89.143, 241.191',transform=axes[2,2].transAxes, 
               fontsize=8,ha='right')
axes[3,2].text(1,0.925,'knots: 50.22, 75.40',transform=axes[3,2].transAxes, 
               fontsize=8,ha='right')

# MaxLST
axes[1,4].text(1,0.925,'knots: 28.780, 33.876',transform=axes[1,4].transAxes, 
               fontsize=8,ha='right')
axes[2,4].text(1,0.925,'knots: 26.709, 35.622',transform=axes[2,4].transAxes, 
               fontsize=8,ha='right')
axes[3,4].text(1,0.925,'knots: 26.709, 35.622',transform=axes[3,4].transAxes, 
               fontsize=8,ha='right')




plt.subplots_adjust(top=0.95, bottom=0.05, left=0.075, right=0.925, hspace=0.3)

# # plt.xticks(range(5,30,2))
# # plt.xticks(range(0,400,25))
plt.savefig('../../visualizationCode/figures/figure_yield_response_lowess.png')
print('figure saved')
