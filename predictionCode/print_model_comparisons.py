import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from soybean_new_model import prediction_result_global

global yield_type_dict,area_type_dict
yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
area_type_dict = {'all': 'area', 'rainfed':'area_rainfed','irrigated':'area_irr'}

# Load the saved predicted results 
def load_prediction_r(model, yield_type='rainfed', prediction_type='forward',state=False, direct_fn=False):
    df = pd.read_csv('%s'%(model), dtype={'FIPS':str})
    # if state:
        # df = pd.read_csv('../lmer_data_files/%s_verification'%(model), dtype={'FIPS':str})
    # else:    
        # df = pd.read_csv('../lmer_data_files/%s_verification'%(model), dtype={'FIPS':str})
    # if direct_fn:
        # df = pd.read_csv('../result/' + direct_fn, dtype={'FIPS':str})

    return df[['year','FIPS','State','Predicted_'+yield_type_dict[yield_type],
           yield_type_dict[yield_type],area_type_dict[yield_type]]]


# Calculate the model prediction performance 
def get_model_performance_r(model, yield_type='rainfed', prediction_type='forward'):
    d = load_prediction_r(model, yield_type=yield_type, prediction_type=prediction_type)
    d_r2 = prediction_result_global(d, yield_type=yield_type)
   # d_r2.loc[:,'rmse']=d_r2.loc[:,'rmse'] * 0.0628 # convert to t/ha
    return d_r2


# Load the saved predicted results 
def load_prediction(model, yield_type='rainfed', prediction_type='forward',state=False, direct_fn=False):

    df = pd.read_csv('%s'%(model), dtype={'FIPS':str})
    # if state:
        # df = pd.read_csv('../result/prediction_%s_%s_%s_state.csv'%(model,yield_type,prediction_type),
            # dtype={'FIPS':str})
    # else:    
        # df = pd.read_csv('../result/prediction_%s_%s_%s.csv'%(model,yield_type,prediction_type),
                    # dtype={'FIPS':str})
    # if direct_fn:
        # df = pd.read_csv('../result/' + direct_fn, dtype={'FIPS':str})

    return df[['year','FIPS','State','Predicted_'+yield_type_dict[yield_type],
           yield_type_dict[yield_type],area_type_dict[yield_type]]]


# Calculate the model prediction performance 
def get_model_performance(model, yield_type='rainfed', prediction_type='forward'):
    d = load_prediction(model, yield_type=yield_type, prediction_type=prediction_type)
    d_r2 = prediction_result_global(d, yield_type=yield_type)
   # d_r2.loc[:,'rmse']=d_r2.loc[:,'rmse'] * 0.0628 # convert to t/ha
    return d_r2

def print_statistics(yield_type='rainfed',prediction_type='forward'):
    # Climate variable selection
    #prediction_type = 'leave_one_year_out'
    
    d0_r2 = get_model_performance('Tgs_linear', yield_type=yield_type,prediction_type=prediction_type)
    d1_r2 = get_model_performance('tave_linear', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('vpd_linear', yield_type=yield_type,prediction_type=prediction_type)
    print('Tgs_linear median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))
    print('tave_linear median R2 and RMSE are %f and %f, respectively'%(d1_r2.median()['R2'],d1_r2.median()['rmse']))
    print('vpd_linear median R2 and RMSE are %f and %f, respectively'%(d2_r2.median()['R2'],d2_r2.median()['rmse']))
    
    
    # Fitting functions 
    d0_r2 = get_model_performance('vpd_linear', yield_type=yield_type,prediction_type=prediction_type)
    d1_r2 = get_model_performance('vpd_poly', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('vpd_spline', yield_type=yield_type,prediction_type=prediction_type)
    print('vpd_linear median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))
    print('vpd_poly median R2 and RMSE are %f and %f, respectively'%(d1_r2.median()['R2'],d1_r2.median()['rmse']))
    print('vpd_spline median R2 and RMSE are %f and %f, respectively'%(d2_r2.median()['R2'],d2_r2.median()['rmse']))
    
    # Satellite data
    d0_r2 = get_model_performance('lstmax_spline_only', yield_type=yield_type,prediction_type=prediction_type)
    d1_r2 = get_model_performance('evi_linear_only', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('evi_poly_only', yield_type=yield_type,prediction_type=prediction_type)
    d3_r2 = get_model_performance('lstmax_spline_evi_poly_only', yield_type=yield_type,prediction_type=prediction_type)

    print('lstmax_spline_only median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))
    print('evi_linear_only median R2 and RMSE are %f and %f, respectively'%(d1_r2.median()['R2'],d1_r2.median()['rmse']))
    print('evi_poly_only median R2 and RMSE are %f and %f, respectively'%(d2_r2.median()['R2'],d2_r2.median()['rmse']))
    print('lstmax_spline_evi_poly_only median R2 and RMSE are %f and %f, respectively'%(d3_r2.median()['R2'],d3_r2.median()['rmse']))
    
    # Satellite data + Climate
    d0_r2 = get_model_performance('vpd_spline_evi', yield_type=yield_type,prediction_type=prediction_type)
    d1_r2 = get_model_performance('tave_spline_evi_poly', yield_type=yield_type,prediction_type=prediction_type)
    d2_r2 = get_model_performance('lstmax_spline_evi_poly', yield_type=yield_type,prediction_type=prediction_type)
    d3_r2 = get_model_performance('vpd_spline_evi_poly', yield_type=yield_type,prediction_type=prediction_type)
    print('vpd_spline_evi median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))
    print('tave_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d1_r2.median()['R2'],d1_r2.median()['rmse']))
    print('lstmax_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d2_r2.median()['R2'],d2_r2.median()['rmse']))
    print('vpd_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d3_r2.median()['R2'],d3_r2.median()['rmse']))
    

if __name__ == '__main__':
    # print_statistics(yield_type='rainfed',prediction_type='leave_one_year_out')

    d0_r2 = get_model_performance('Tgs_linear', yield_type="rainfed",prediction_type="leave_one_year_out")
    print('Corn_percent at 0.001')
    print('Python: vpd_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))

    d0_r2 = get_model_performance_r('Tgs_linear', yield_type="rainfed",prediction_type="leave_one_year_out")
    print('R: vpd_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))
    print()

    d0_r2 = get_model_performance('vpd_spline_evi_poly', yield_type="rainfed",prediction_type="leave_one_year_out")
    print('Corn_percent at 0.001')
    print('Python: vpd_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))


    d0_r2 = get_model_performance_r('vpd_spline_evi_poly', yield_type="rainfed",prediction_type="leave_one_year_out")
    print('R: vpd_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))
    print()


    print('Corn_percent at 0.003')
    d0_r2 = get_model_performance('lstmax_spline_evi_poly', yield_type="rainfed",prediction_type="leave_one_year_out")
    print('Python: lstmax_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))


    
    d0_r2 = get_model_performance_r('lstmax_spline_evi_poly', yield_type="rainfed",prediction_type="leave_one_year_out")
    print('R: lstmax_spline_evi_poly median R2 and RMSE are %f and %f, respectively'%(d0_r2.median()['R2'],d0_r2.median()['rmse']))

#    d0_r2 = get_model_performance('vpd_spline', yield_type='rainfed',prediction_type='leave_one_year_out')
#    d1_r2 = get_model_performance('vpd_spline_evi_poly', yield_type='rainfed',prediction_type='leave_one_year_out')
