#:cd %:h
#head 

import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
lowess = sm.nonparametric.lowess


# Z = lowess(data[con]['yield_ana'], data[con][x_var],frac=0.3,it=3)
# plt.plot(Z[:,0], Z[:,1], 'g-', lw=5)
# plt.scatter(data[con][x_var], data[con]['yield_ana'])

def yield_trend(df, yield_type='rainfed'):
    """
    Get a fitted trend function, in order to de-trend yield.
    :param df: A data frame.
    :param yield_type: The yield type to detrend.
    :return: A fitted trend function.
    """
    yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
    # Estimate regional yield trend and detrend
    trend_model_txt = "Q('%s')"%yield_type_dict[yield_type] + "~ year"
    trend_results = smf.ols(trend_model_txt, data=df).fit()
    return trend_results


def load_yield_data():
    """
    Take the initial raw data frame and preprocess it.
    :return: A preprocessed data frame.
    """
    data = pd.read_csv("soybean_model_data_2017.csv",dtype={'FIPS':str})
    
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
    
    return data_12

def printOffByOne():
    """
    Print lowess curves for each model one at a time.
    :return: Nothing.
    """
    data = load_yield_data()
    numberColumns = 5
    firstEntry = 'tmax5'
    lastEntry = 'lstmax9'
    colNames = list(data)
    firstIndex =colNames.index(firstEntry)
    lastIndex = colNames.index(lastEntry)
    numberTypesOfVariables = 5
    months = 5
    variables = ['tave5', 'tave6', 'tave7', 'tave8', 'tave9', 'vpdave5', 'vpdave6', 'vpdave7', 'vpdave8', 'vpdave9', 'precip5', 'precip6', 'precip7', 'precip8', 'precip9', 'evi5', 'evi6', 'evi7', 'evi8', 'evi9', 'lstmax5', 'lstmax6', 'lstmax7', 'lstmax8', 'lstmax9']
    variables = ['tave5', 'tave6', 'tave7', 'tave8', 'tave9', 'vpdave5', 'vpdave6', 'vpdave7', 'vpdave8', 'vpdave9', 'precip5', 'precip6', 'precip7', 'precip8', 'precip9', 'evi5', 'evi6', 'evi7', 'evi8', 'evi9', 'lstmax5', 'lstmax6', 'lstmax7', 'lstmax8', 'lstmax9']
    print(firstIndex, lastIndex)
    print(colNames)
    for i in range(len(variables)):
        plt.plot(data[variables[i]], data["yield_rainfed_ana"],'bx')
        plt.title([variables[i]])
        Z = lowess(data['yield_rainfed_ana'], data[variables[i]],frac=0.3,it=3)
        plt.plot(Z[:,0], Z[:,1], 'g-', lw=5)
        plt.title("Response for %s"%(variables[i]))
        plt.show()

def printAll():
    """
    Print lowess curves for all models.
    :return: Nothing.
    """
    data = load_yield_data()
    numberColumns = 5
    firstEntry = 'tmax5'
    lastEntry = 'lstmax9'
    colNames = list(data)
    firstIndex =colNames.index(firstEntry)
    lastIndex = colNames.index(lastEntry)
    numberTypesOfVariables = 5
    months = 5
    f, axarr = plt.subplots(numberTypesOfVariables, months)
    variables = ['tave5', 'tave6', 'tave7', 'tave8', 'tave9', 'vpdave5', 'vpdave6', 'vpdave7', 'vpdave8', 'vpdave9', 'precip5', 'precip6', 'precip7', 'precip8', 'precip9', 'evi5', 'evi6', 'evi7', 'evi8', 'evi9', 'lstmax5', 'lstmax6', 'lstmax7', 'lstmax8', 'lstmax9']
    print(firstIndex, lastIndex)
    print(colNames)
    for i in range(len(variables)):
            axarr[int(i/numberColumns), int(i%numberColumns)].plot(data[variables[i]], data["yield_rainfed_ana"],'bx')
            axarr[int(i/numberColumns), int(i%numberColumns)].set_title([variables[i]])
            Z = lowess(data['yield_rainfed_ana'], data[variables[i]],frac=0.3,it=3)
            axarr[int(i/numberColumns), int(i%numberColumns)].plot(Z[:,0], Z[:,1], 'g-', lw=5)
    plt.show()

if __name__ == "__main__":
    printOffByOne()

