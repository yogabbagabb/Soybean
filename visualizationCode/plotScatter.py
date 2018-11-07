import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import os
sys.path.append('../predictionCode')
from soybean_new_model import *



if __name__ == "__main__":

    # Change our working directory. After changing it, we can read
    # the CSVs we need.
    os.chdir('../dataFiles/newR_Prediction_CSVs')
    A = pd.read_csv("vpd_spline_evi_poly")
    os.chdir('./statsDirectory/')
    C = pd.read_csv("vpd_spline_evi_poly_stats.csv")
# trend_function = yield_trend(A)
    # A['yield_rainfed'] = A['yield_rainfed_ana'] + trend_function.predict(A)

    os.chdir('../../../visualizationCode')
    for year in range(2003,2017,1):
        B = A[A['year'] == year]
        B = B[~pd.isnull(B["yield_rainfed"])]
        print(year)

        model_txt = "yield_rainfed ~ Predicted_yield_rainfed"
        func = smf.ols(model_txt, data = B, missing='drop').fit()
        plt.scatter(B["Predicted_yield_rainfed"],B["yield_rainfed"],label=None)
        plt.xlim((0,70))
        plt.ylim((0,70))
        straight = [i for i in range (0,71,1)]
        plt.plot(straight, straight,'g-', label = "y = x")
        straightDf = pd.DataFrame(straight, columns = ['Predicted_yield_rainfed'])
        plt.plot(straight, func.predict(straightDf), 'r-',label='Predicted Yield vs. Actual Yield Best Fit')
        number = year - 2003
        firstString = "RMSE: %s"%(C.loc[(number),'rmse'])
        secondString = "R2 %s"%(C.loc[(number),'R2'])
        print(firstString)
        print(secondString)
        plt.plot([],[], ' ',  label = firstString)
        plt.plot([],[], ' ',  label = secondString)
        plt.legend(loc=2)
        plt.xlabel("Predicted yield rainfed (bushels/acre)")
        plt.ylabel("Actual yield rainfed (bushels/acre)")
        plt.title("Year %s"%year)
        csw = os.getcwd()
        print(csw)
        plt.savefig('figures/scatterPlots/scatter_plot_vpd_spline_evi_poly_%s'%year)
        plt.clf()

