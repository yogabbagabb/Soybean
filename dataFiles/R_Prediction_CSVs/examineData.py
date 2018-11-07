import pandas as pd
from func_crop_model import *

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
    A = pd.read_csv("soybean_model_data_2017.csv")
    trend_function = yield_trend(A)
    for aFile in csvFiles:
        B = pd.read_csv(aFile)
        B.rename(index=str,columns={"prediction..j..":"Predicted_yield_rainfed"})
        B["Predicted_yield_rainfed"] += trend_function.predict(B)
        B.to_csv(aFile + "_Yan_suitable.csv")



if __name__ == "__main__":
    manipulate_R_CSV(["vpd_spline_evi_poly"])
