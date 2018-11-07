
var_base = {'T_linear_6_8': " + tave6 + tave7 + tave8",
            
            'P_linear_6_9': " + precip6 + precip7 + precip8 + precip9",
            
            'VPD_linear_6_8': " + vpdave6 + vpdave7 + vpdave8",
            
            'T_poly_6_8': (" + tave6 + tave7 + tave8"  
                           "+ np.power(tave6, 2) + np.power(tave7, 2) + np.power(tave8, 2)"),
            
            'P_poly_6_9': (" + precip6 + precip7 + precip8 + precip9" +
                           "+ np.power(precip6, 2) + np.power(precip7, 2) + np.power(precip8, 2) + np.power(precip9, 2)"),
            
            'VPD_poly_6_8': ("+ vpdave6 + vpdave7 + vpdave8" + 
                             " + np.power(vpdave6, 2) + np.power(vpdave7, 2) + np.power(vpdave8, 2)"),
            
            'LSTMAX_linear_6_8': "+ lstmax6 + lstmax7 + lstmax8",
            
            'LSTMAX_poly_6_8': ("+ lstmax6 + lstmax7 + lstmax8" + 
                                "+ np.power(lstmax6, 2) + np.power(lstmax7, 2) + np.power(lstmax8, 2)"),
            
            'T_spline_6_8': ("+ bs(tave6, df=3, knots = (20.6,22.6), degree=1,lower_bound=7,upper_bound=35)"
                            + "+ bs(tave7, df=3, knots = (22.5,25.5), degree=1,lower_bound=10,upper_bound=40)" 
                            + "+ bs(tave8, df=3, knots = (21.5,23.5), degree=1,lower_bound=11,upper_bound=40)"), 
                                
            'VPD_spline_6_8': ("+ bs(vpdave6, df=4, knots = (8.5,10.5,12.5), degree=1,lower_bound=4,upper_bound=30)" 
                              + "+ bs(vpdave7, df=3, knots = (8,10.5), degree=1,lower_bound=4,upper_bound=35)" 
                              + " + bs(vpdave8, df=3, knots = (8.06,9.10), degree=1,lower_bound=3,upper_bound=30)"),
                                
            'P_spline_6_9': (" + bs(precip6, df=3, knots = (92.6,182.2), degree=1,lower_bound=0,upper_bound=500)"
                            + " + bs(precip7, df=3, knots = (56.191,89.1428), degree=1,lower_bound=0,upper_bound=600)"
                            + " + bs(precip8, df=3, knots = (50.22, 75.40), degree=1,lower_bound=0,upper_bound=500)"
                            + " + bs(precip9, df=4, knots = (29.9037,61,104.1), degree=1, lower_bound=0, upper_bound=500)"),

            
            'LSTMAX_spline_6_8':("+ bs(lstmax6, df=4, knots = (28.6, 33.1, 34.5), degree=1,lower_bound=20,upper_bound=50)" 
                                + "+ bs(lstmax7, df=3, knots = (26.4,34.5), degree=1,lower_bound=20,upper_bound=53)" 
                                + "+ bs(lstmax8, df=4, knots = (26.1,28.1,29.3), degree=1, lower_bound=18, upper_bound=48)"),
            
            'EVI_linear_5_9': "+ evi5 + evi6 + evi7 + evi8 + evi9",
            
            'EVI_poly_5_9': ("+ evi5 + evi6 + evi7 + evi8 + evi9"
                             + " + np.power(evi5,2) + np.power(evi6, 2) + np.power(evi7, 2) + np.power(evi8, 2)"
                             + " + np.power(evi9, 2)"),
            
            'EVI_spline_5_9': (" + bs(evi5, df=4, knots= (0.20,0.25,0.33), degree=1,upper_bound=0.8)"
                             + " + bs(evi6, df=3, knots= (0.25,0.35), degree=1,upper_bound=0.8)"
                             + " + bs(evi7, df=3, knots= (0.22,0.61), degree=1,upper_bound=0.8)"
                             + " + bs(evi8, df=3, knots= (0.58,0.63), degree=1,upper_bound=0.8)"
                             + " + bs(evi9,df=3, knots= (0.35,0.51), degree=1,upper_bound=0.8)"),

            'Tgs_linear': (" + tave56789 + precip56789"), 
            
            'Tgs_poly': (" + tave56789 + np.power(tave56789, 2)"
                         + " + precip56789 + np.power(precip56789, 2)")         
           }

def getModelStructure(name, yield_type='all'):
    yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
    
    if name == 'tave_linear':
        model_vars = var_base['T_linear_6_8'] + var_base['P_linear_6_9']

    if name == 'Tgs_linear':
        model_vars = var_base['Tgs_linear']
    
    if name == 'tave_spline':
        model_vars = var_base['T_spline_6_8'] + var_base['P_spline_6_9']
        
    if name == 'tave_poly':
        model_vars = var_base['T_poly_6_8'] + var_base['P_poly_6_9']

    if name == 'Tgs_poly':
        model_vars = var_base['Tgs_poly']
        
    if name == 'vpd_linear':
        model_vars = var_base['VPD_linear_6_8'] + var_base['P_linear_6_9']
        
    if name == 'vpd_poly':
        model_vars = var_base['VPD_poly_6_8'] + var_base['P_poly_6_9']    
        
    if name == 'vpd_spline':
        model_vars = var_base['VPD_spline_6_8'] + var_base['P_spline_6_9']       

    if name == 'lstmax_linear_only':
        model_vars = var_base['LSTMAX_linear_6_8']
        
    if name == 'lstmax_poly_only':
        model_vars = var_base['LSTMAX_poly_6_8']
        
    if name == 'lstmax_spline_only':
        model_vars = var_base['LSTMAX_spline_6_8']   
        
    if name == 'evi_linear_only':
        model_vars = var_base['EVI_linear_5_9']  
        
    if name == 'evi_poly_only':
        model_vars = var_base['EVI_poly_5_9']     
        
    if name == 'evi_spline_only':
        model_vars = var_base['EVI_spline_5_9']    
        
    if name == 'tave_spline_evi':
        model_vars =var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_linear_5_9'] 
        
    if name == 'tave_spline_evi_poly':
        model_vars =var_base['T_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5_9'] 
        
    if name == 'tave_poly_evi':
        model_vars =var_base['T_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_linear_5_9']      
        
    if name == 'vpd_spline_evi':
        model_vars =var_base['VPD_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_linear_5_9']  
        
    if name == 'vpd_poly_evi':
        model_vars =var_base['VPD_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_linear_5_9']   
        
    if name == 'vpd_poly_evi_poly':
        model_vars =var_base['VPD_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_poly_5_9']    
        
    if name == 'vpd_spline_evi_poly':
        model_vars =var_base['VPD_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5_9']         
        
    if name == 'vpd_spline_evi_poly_aahan':
        model_vars =var_base['VPD_spline_6_8'] + var_base['P_spline_6_9_aahan'] + var_base['EVI_poly_5_9']         

    if name == 'lstmax_spline_evi':
        model_vars = var_base['LSTMAX_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_linear_5_9'] 
        
    if name == 'lstmax_poly_evi_poly':
        model_vars = var_base['LSTMAX_poly_6_8'] + var_base['P_poly_6_9'] + var_base['EVI_poly_5_9']

    if name == 'lstmax_poly_evi_poly_only':
        model_vars = var_base['LSTMAX_poly_6_8'] + var_base['EVI_poly_5_9']
        
    if name == 'lstmax_spline_evi_poly':
        model_vars = var_base['LSTMAX_spline_6_8'] + var_base['P_spline_6_9'] + var_base['EVI_poly_5_9'] 
        
    if name == 'lstmax_spline_evi_poly_only':
        model_vars = var_base['LSTMAX_spline_6_8'] + var_base['EVI_poly_5_9']         
        
    return ("Q('%s_ana') ~ "%yield_type_dict[yield_type] + model_vars + "+ C(FIPS)")

if __name__ == "__main__":

    models = ['Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly',#4 
              ]
    for m in models:
        modelText = getModelStructure(m,yield_type="rainfed")
        print("\n"+modelText+"\n")
