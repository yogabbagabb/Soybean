
    yield_prediction_csv_1 <- "./all_yan_models_prediction"
    rmse_csv_1 <- "./all_yan_models_rmse"

    model_formulas_1 <- c("yield_rainfed_ana ~ tave56789 + precip56789+ FIPS",

"yield_rainfed_ana ~ tave56789 + I(tave56789^2) + precip56789 + I(precip56789^2)+ FIPS",

"yield_rainfed_ana ~ tave6 + tave7 + tave8 + precip6 + precip7 + precip8 + precip9+ FIPS",

"yield_rainfed_ana ~ vpdave6 + vpdave7 + vpdave8 + precip6 + precip7 + precip8 + precip9+ FIPS",

"yield_rainfed_ana ~ tave6 + tave7 + tave8+ I(tave6^2) + I(tave7^2) + I(tave8^2) + precip6 + precip7 + precip8 + precip9+ I(precip6^2) + I(precip7^2) + I(precip8^2) + I(precip9^2)+ FIPS",

"yield_rainfed_ana ~ vpdave6 + vpdave7 + vpdave8 + I(vpdave6^2) + I(vpdave7^2) + I(vpdave8^2) + precip6 + precip7 + precip8 + precip9+ I(precip6^2) + I(precip7^2) + I(precip8^2) + I(precip9^2)+ FIPS",

"yield_rainfed_ana ~ lstmax6 + lstmax7 + lstmax8+ FIPS",

"yield_rainfed_ana ~ lstmax6 + lstmax7 + lstmax8+ I(lstmax6^2) + I(lstmax7^2) + I(lstmax8^2)+ FIPS",

"yield_rainfed_ana ~ evi5 + evi6 + evi7 + evi8 + evi9+ FIPS",

"yield_rainfed_ana ~ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ lstmax6 + lstmax7 + lstmax8+ I(lstmax6^2) + I(lstmax7^2) + I(lstmax8^2)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ vpdave6 + vpdave7 + vpdave8 + I(vpdave6^2) + I(vpdave7^2) + I(vpdave8^2) + precip6 + precip7 + precip8 + precip9+ I(precip6^2) + I(precip7^2) + I(precip8^2) + I(precip9^2)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(evi5,  degree=1, knots = c(0.20,0.25,0.33),df=4) + bs(evi6,  degree=1, knots = c(0.25,0.35),df=3) + bs(evi7,  degree=1, knots = c(0.22,0.61),df=3) + bs(evi8,  degree=1, knots = c(0.58,0.63),df=3) + bs(evi9, degree=1, knots = c(0.35,0.51),df=3)+ FIPS",

"yield_rainfed_ana ~ bs(lstmax6,  degree=1, knots = c(28.6, 33.1, 34.5),df=4)+ bs(lstmax7,  degree=1, knots = c(26.4,34.5),df=3)+ bs(lstmax8,  degree=1, knots = c(26.1,28.1,29.3),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(lstmax6,  degree=1, knots = c(28.6, 33.1, 34.5),df=4)+ bs(lstmax7,  degree=1, knots = c(26.4,34.5),df=3)+ bs(lstmax8,  degree=1, knots = c(26.1,28.1,29.3),df=4) + bs(precip6,  degree=1, knots = c(92.6,182.2),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,61,104.1),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(tave6,  degree=1, knots = c(20.6,22.6),df=3)+ bs(tave7,  degree=1, knots = c(22.5,25.5),df=3)+ bs(tave8,  degree=1, knots = c(21.5,23.5),df=3) + bs(precip6,  degree=1, knots = c(92.6,182.2),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,61,104.1),df=4)+ FIPS",

"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.552),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ FIPS",

"yield_rainfed_ana ~ bs(lstmax6,  degree=1, knots = c(28.780, 33.876),df=3)+ bs(lstmax7,  degree=1, knots = c(26.709,35.622),df=3)+ bs(lstmax8,  degree=1, knots = c(25.901,29.033),df=3)+ FIPS",

"yield_rainfed_ana ~ bs(tave6,  degree=1, knots = c(21.014, 23.182),df=3)+ bs(tave7,  degree=1, knots = c(22.537,25.448),df=3)+ bs(tave8,  degree=1, knots = c(21.55,24.815),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9+ FIPS",

"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5,14.96), df=5)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.552),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9+ FIPS",

"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.5521),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.096),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

#"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,9.10),df=3) + bs(precip6,  degree=1, knots = c(92.6,182.2),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,61,104.1),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(tave6,  degree=1, knots = c(21.014,23.182),df=3)+ bs(tave7,  degree=1, knots = c(22.537,25.448),df=3)+ bs(tave8,  degree=1, knots = c(21.55,24.815),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS")




    model_names_1 <- c('Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly')#4 
    
    fitting_functions_1 <- c("lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm")
    svd_issue_1 <- c("N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N")
    uses_FIPS_1 <- c("Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y")
    uses_evi_1 <- c("N","N","N","N","N","N","N","N","Y","Y","Y","Y","Y","Y","Y","N","N","N","Y","Y","Y","Y")
    uses_lstmax_1 <- c("N","N","N","N","N","N","Y","Y","N","N","Y","N","N","Y","Y","N","N","Y","N","N","N","N")

#______________
    yield_prediction_csv_1 <- "./vpd_spline_evi_poly_soybean_prediction"
    rmse_csv_1 <- "./vpd_spline_evi_poly_soybean_rmse"

    model_formulas_1 <- c("yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5,14.96),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.5521),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.096),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS")





    model_names_1 <- c('vpd_spline_evi_poly_2')

    fitting_functions_1 <- c("lm")
    svd_issue_1 <- c("N")
    uses_FIPS_1 <- c("Y")
    uses_evi_1 <- c("Y")
    uses_lstmax_1 <- c("N")

#______________
    yield_prediction_csv_1 <- "./vpd_spline_evi_poly_soybean_prediction"
    rmse_csv_1 <- "./vpd_spline_evi_poly_soybean_rmse"

    model_formulas_1 <- c("yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5,14.96),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.5521),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.096),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS")





    model_names_1 <- c('vpd_spline_evi_poly_2')

    fitting_functions_1 <- c("lm")
    svd_issue_1 <- c("N")
    uses_FIPS_1 <- c("Y")
    uses_evi_1 <- c("Y")
    uses_lstmax_1 <- c("N")


