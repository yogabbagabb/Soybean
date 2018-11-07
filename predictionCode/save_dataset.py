import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

def yield_trend(df, yield_type='rainfed'):
    yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
    # Estimate regional yield trend and detrend
    trend_model_txt = "Q('%s')"%yield_type_dict[yield_type] + "~ year"
    trend_results = smf.ols(trend_model_txt, data=df).fit()
    return trend_results

def load_yield_data():
    data = pd.read_csv('./soybean_model_data_2017.csv',dtype={'FIPS':str})
    
    data['soybean_percent'] = data['area']/data['land_area']
    
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

    # add growing season
    data['tave56789']= data.loc[:,'tave5':'tave9'].mean(axis=1)
    data['vpdave56789']= data.loc[:,'vpdave5':'vpdave8'].mean(axis=1)
    data['precip56789']= data.loc[:,'precip5':'precip9'].sum(axis=1)
    
    
    # Add z-score
    county_std = data.groupby('FIPS').std()['precip56789'].to_frame('precip_gs_std').reset_index()
    county_mean = data.groupby('FIPS').mean()['precip56789'].to_frame('precip_gs_mean').reset_index()
    
    data = data.merge(county_mean, on='FIPS').merge(county_std, on='FIPS')
    
    data['precip_gs_z'] = (data['precip56789'] - data['precip_gs_mean'])/data['precip_gs_std']

    # The 12 core states 
    data_12 = data[data['State'].isin(data.loc[data['evi6'].notnull(),'State'].unique())]

    # Detrend yield
    global trend_rainfed, trend_irrigated, trend_all
    trend_rainfed = yield_trend(data_12, yield_type='rainfed')
    trend_irrigated = yield_trend(data_12, yield_type='irrigated')
    trend_all = yield_trend(data_12, yield_type='all')
    
    data_12.loc[:,'yield_ana'] = (data_12['yield'] - trend_all.predict(data_12[['year','yield']]))
    data_12.loc[:,'yield_rainfed_ana'] = (data_12['yield_rainfed'] - trend_rainfed.predict(data_12[['year','yield_rainfed']]))      
    data_12.loc[:,'yield_irr_ana'] = (data_12['yield_irr'] - trend_irrigated.predict(data_12[['year','yield_irr']])) 
    
    data_12.to_csv('./soybean_handled_dataset')

if __name__ == "__main__":
    load_yield_data()

