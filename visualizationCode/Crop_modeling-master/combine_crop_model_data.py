import pandas as pd

"""
Load soil variable 
"""
def get_soil_data(var):
    
    soil1 = pd.read_csv('~/Project/data/GEE/US_soil_corn_county/zonal_corn_%s_mean_0_5.csv'%var,
                header=0, names=['FIPS', 'mean'],usecols=[1,2])
    soil2 = pd.read_csv('~/Project/data/GEE/US_soil_corn_county/zonal_corn_%s_mean_5_15.csv'%var,
                header=0, names=['FIPS', 'mean'],usecols=[1,2])
    soil3 = pd.read_csv('~/Project/data/GEE/US_soil_corn_county/zonal_corn_%s_mean_15_30.csv'%var,
                header=0, names=['FIPS', 'mean'],usecols=[1,2])
    
    soil1[var] = soil1['mean']/6 +   soil2['mean']/3 + soil3['mean']/2
    
    soil1.dropna(subset=['FIPS'], inplace=True)

    soil1['FIPS'] = soil1['FIPS'].map(lambda x:"%05d"%(x))
    
    return soil1[['FIPS',var]]
    

nass = pd.read_csv('../data/nass_yield_area_1981_2016.csv',dtype={'FIPS':str})

climate = pd.read_csv('../data/prism_climate_growing_season_1981_2016.csv',dtype={'FIPS':str})

evi = pd.read_csv('../data/US_county_corn_evi_2000-2016.csv',dtype={'FIPS':str})

lst = pd.read_csv('../data/US_county_corn_lstmax_2003-2016.csv',dtype={'FIPS':str})

om = get_soil_data('om')
awc = get_soil_data('awc')

# county area
county = pd.read_csv('~/Project/data/US_county_gis/counties.csv',dtype={'FIPS':str})
county.rename(columns={'AREA':'land_area'}, inplace=True)
county['land_area'] = county['land_area'] * 640 # mi2 to acers

df_final=nass.merge(climate,on=['year','FIPS'],how='left')\
              .merge(evi,on=['year','FIPS'],how='left')\
              .merge(lst,on=['year','FIPS'],how='left')\
              .merge(om,on=['FIPS'],how='left')\
              .merge(awc,on=['FIPS'],how='left')\
              .merge(county[['FIPS','land_area']],on=['FIPS'],how='left')

df_final.to_csv('../data/Corn_model_data.csv',index=False)
print('Crop model data csv file saved')
