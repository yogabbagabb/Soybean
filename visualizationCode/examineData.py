import numpy as np
import pandas as pd
import sys
sys.path.insert(0,'../predictionCode/')
import statsmodels.formula.api as smf
import statsmodels.api as sm
from soybean_new_model import *
from matplotlib import pyplot as plt
import os


homePath = os.getcwd()
def examineData():
    A = pd.read_csv("soybean_model_data_2017.csv")

    # Test if precipX, tempX and vpdaveX and lstmaxX are each missing (ie NA) iff the remaining three entries are missing (ie NA)
    first = pd.isnull(A["precip5"]) & pd.isnull(A["precip6"]) & pd.isnull(A["precip7"]) & pd.isnull(A["precip8"]) & pd.isnull(A["precip9"])
    second = ~(pd.isnull(A["precip5"])) & ~(pd.isnull(A["precip6"])) & ~(pd.isnull(A["precip7"])) & ~(pd.isnull(A["precip8"])) & ~(pd.isnull(A["precip9"]))
    print(any(~(first | second)))


    first = pd.isnull(A["vpdave5"]) & pd.isnull(A["vpdave6"]) & pd.isnull(A["vpdave7"]) & pd.isnull(A["vpdave8"]) & pd.isnull(A["vpdave9"])
    second = ~(pd.isnull(A["vpdave5"])) & ~(pd.isnull(A["vpdave6"])) & ~(pd.isnull(A["vpdave7"])) & ~(pd.isnull(A["vpdave8"])) & ~(pd.isnull(A["vpdave9"]))
    print(any(~(first | second)))

    first = pd.isnull(A["lstmax5"]) & pd.isnull(A["lstmax6"]) & pd.isnull(A["lstmax7"]) & pd.isnull(A["lstmax8"]) & pd.isnull(A["lstmax9"])
    second = ~(pd.isnull(A["lstmax5"])) & ~(pd.isnull(A["lstmax6"])) & ~(pd.isnull(A["lstmax7"])) & ~(pd.isnull(A["lstmax8"])) & ~(pd.isnull(A["lstmax9"]))
    print(any(~(first | second)))

    first = pd.isnull(A["evi5"]) & pd.isnull(A["evi6"]) & pd.isnull(A["evi7"]) & pd.isnull(A["evi8"]) & pd.isnull(A["evi9"])
    second = ~(pd.isnull(A["evi5"])) & ~(pd.isnull(A["evi6"])) & ~(pd.isnull(A["evi7"])) & ~(pd.isnull(A["evi8"])) & ~(pd.isnull(A["evi9"]))
    print(any(~(first | second)))

    first = pd.isnull(A["tave5"]) & pd.isnull(A["tave6"]) & pd.isnull(A["tave7"]) & pd.isnull(A["tave8"]) & pd.isnull(A["tave9"])
    second = ~(pd.isnull(A["tave5"])) & ~(pd.isnull(A["tave6"])) & ~(pd.isnull(A["tave7"])) & ~(pd.isnull(A["tave8"])) & ~(pd.isnull(A["tave9"]))
    print(any(~(first | second)))

    first = pd.isnull(A["lstmax5"]) & pd.isnull(A["evi5"]) 
    second = ~(pd.isnull(A["lstmax5"])) & ~(pd.isnull(A["evi5"]))
    print(any(~(first | second)))


    first = pd.isnull(A["tave5"]) & ~pd.isnull(A["lstmax5"])
    print(any(first))

    first = pd.isnull(A["tave5"]) & ~pd.isnull(A["evi5"])
    print(any(first))


def manipulate_R_CSV(csvFiles):
    os.chdir("../dataFiles")
    A = pd.read_csv("soybean_handled_dataset")
    os.chdir(homePath)
    trend_function = yield_trend(A)
    for aFile in csvFiles:
        os.chdir("../dataFiles/newR_Prediction_CSVs")
        B = pd.read_csv(aFile)
        B = B.rename(index=str,columns={"predictions..j..":"Predicted_yield_rainfed"})
        B["Predicted_yield_rainfed"] += trend_function.predict(B)
        # B.to_csv(aFile + "_Yan_suitable.csv")
        B.to_csv(aFile)
        os.chdir(homePath)

def printPlots(predicNames):
    A = pd.read_csv("../dataFiles/soybean_handled_dataset")
    for predic in predicNames:
        B = A[~pd.isnull(A[predic])]
        xArr = np.linspace(B[predic].min(), B[predic].max(), 100)
        xArr = pd.DataFrame(xArr)
        xArr.columns = [predic]
        print(xArr)
        rootName = "/Users/anjaliagrawal/Documents/Aahan/UIUC/Research/crop_yield/Soybean/Crop_modeling-master/figure/Responses/"
        plt.ylabel("yield_rainfed_ana (bushels per acre)")

        plt.plot(B[predic], B["yield_rainfed_ana"], 'bo')
        trend_model_txt = "yield_rainfed_ana ~ " + predic
        trend_results = smf.ols(trend_model_txt, data=B).fit()
        plt.plot(xArr, trend_results.predict(xArr))
        # plt.plot(B[predic], trend_results.predict(B[predic]))
        plt.xlabel(predic + "_linear")
        plt.savefig(rootName + predic + "_linear.png")
        plt.clf()


        # xArr = np.linspace((B[predic]**2).min(), (B[predic]**2).max(), 100)
        # xArr = pd.DataFrame(xArr)
        # xArr.columns = [predic]
        plt.plot(B[predic]**2, B["yield_rainfed_ana"], 'bo')
        trend_model_txt = "yield_rainfed_ana ~ " + "np.power(" + predic + ",2)"
        trend_results = smf.ols(trend_model_txt, data=B).fit()
        plt.plot(xArr**2, trend_results.predict(xArr))
        # plt.plot(B[predic]**2, trend_results.predict(B[predic]**2))
        plt.xlabel(predic + "_quadratic")
        plt.ylabel("yield_rainfed_ana (bushels per acre)")
        plt.savefig(rootName + predic + "_squared.png")
        plt.clf()


        # xArr = np.linspace((B[predic]**3).min(), (B[predic]**3).max(), 100)
        # xArr = pd.DataFrame(xArr)
        # xArr.columns = [predic]
        plt.plot(B[predic]**3, B["yield_rainfed_ana"], 'bo')
        trend_model_txt = "yield_rainfed_ana ~ " + "np.power(" + predic + ",3)"
        trend_results = smf.ols(trend_model_txt, data=B).fit()
        plt.plot(xArr**3, trend_results.predict(xArr))
        # plt.plot(B[predic]**3, trend_results.predict(B[predic]**3))
        plt.xlabel(predic + "_cubic")
        plt.ylabel("yield_rainfed_ana (bushels per acre)")
        plt.savefig(rootName + predic + "_cubed.png")
        plt.clf()



if __name__ == "__main__":
    manipulate_R_CSV(["vpd_spline_evi_poly"])
