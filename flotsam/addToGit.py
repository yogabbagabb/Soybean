import os

if __name__ == "__main__":
    models = ['Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly',#4
              ]
    for m in models:
        command = "git add " + m
        os.system(command)
