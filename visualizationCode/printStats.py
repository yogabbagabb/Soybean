import numpy as np
import pandas as pd
import os

homePath = os.getcwd()


# After specifying the rmse csv files, this simple function calculates the median for each
# set of files
def printRMSEMedian():
    # csvFiles = ["vpd_tave_rmse.csv", "precip_evi_rmse.csv", "tave_evi_rmse.csv",
            # "vpd_evi_rmse.csv", "tave_precip_rmse.csv", "vpd_precip_rmse.csv"]
    # csvFiles = ["vpd_evi_tave_nested_rmse_loo_square.csv","vpd_evi_tave_nested_rmse_loo_square_2.csv", "tave_vpd_evi_rmse_loo_square.csv"]
    csvFiles = ["tave_evi_squared_rmse.csv"]

    for file in csvFiles:
        A = pd.read_csv(file)
        A = A.iloc[:,1:]
        print(A.median())

def calculateR2(csvFiles):
    indexOfFirstModel = 7
    for aFile in csvFiles:
        modelFrame = pd.read_csv(aFile)
        # Get the column names from the data frame
        colNames = list(modelFrame)
        for model in colNames[indexOfFirstModel::]:

            result = pd.DataFrame(np.full([modelFrame['year'].unique().shape[0],3], np.nan), index=modelFrame['year'].unique(), columns=['R2','rmse','R2_classic'])

            for y in range(modelFrame['year'].min(), modelFrame['year'].max()+1):
                con = modelFrame['year']==y
                yieldType = "yield_rainfed"
                r2_temp = modelFrame.loc[con,[yieldType, \
                                        model]].corr() \
                    [model][0]**2
                
                # N is the sample number after removing nan
                N = modelFrame.loc[con,[yieldType,model]].dropna().shape[0]
                rmse_temp = (((modelFrame.loc[con, model] -  \
                                  modelFrame.loc[con, yieldType])**2).sum() \
                                              /N)**0.5
                                          #    /modelFrame.loc[con,yieldType].shape[0])**0.5
                             
        #                                       /modelFrame.loc[con,model].shape[0])**0.5

                sst = ((modelFrame.loc[con, yieldType] \
                        - modelFrame.loc[con, yieldType].mean())**2).sum()
                sse = ((modelFrame.loc[con, yieldType] - modelFrame.loc[con, model])**2).sum()

                result.loc[y] = [r2_temp, rmse_temp, 1-sse/sst]
            print(model)
            # print(result)
            print(result.median()['rmse'])
            print(result.median()['R2'])

            result.to_csv("vpd_spline_evi_poly_2_real_stats.csv")


def calculateR2Emergency(csvFiles):
    indexOfFirstModel = 1
    for aFile in csvFiles:
        modelFrame = pd.read_csv(aFile)
        # Get the column names from the data frame
        colNames = list(modelFrame)
        model = colNames[indexOfFirstModel]

        result = pd.DataFrame(np.full([modelFrame['year'].unique().shape[0],3], np.nan), index=modelFrame['year'].unique(), columns=['R2','rmse','R2_classic'])

        for y in range(modelFrame['year'].min(), modelFrame['year'].max()+1):
            con = modelFrame['year']==y
            yieldType = "yield_rainfed_ana"
            r2_temp = modelFrame.loc[con,[yieldType, \
                                    model]].corr() \
                [model][0]**2
            
            # N is the sample number after removing nan
            N = modelFrame.loc[con,[yieldType,model]].dropna().shape[0]
            rmse_temp = (((modelFrame.loc[con, model] -  \
                              modelFrame.loc[con, yieldType])**2).sum() \
                                          /N)**0.5
                                      #    /modelFrame.loc[con,yieldType].shape[0])**0.5
                         
    #                                       /modelFrame.loc[con,model].shape[0])**0.5

            sst = ((modelFrame.loc[con, yieldType] \
                    - modelFrame.loc[con, yieldType].mean())**2).sum()
            sse = ((modelFrame.loc[con, yieldType] - modelFrame.loc[con, model])**2).sum()

            result.loc[y] = [r2_temp, rmse_temp, 1-sse/sst]
        print(model)
        # print(result)
        print(result)
        statsCSV = "./statsDirectory/" + str(aFile) + "_stats.csv"
        result.to_csv(statsCSV)
        print(result.median()['rmse'])
        print(result.median()['R2'])

def calculateR2EmergencyYan(csvFiles):
    for aFile in csvFiles:
        modelFrame = pd.read_csv(aFile)
        # Get the column names from the data frame
        colNames = list(modelFrame)
        model = "Predicted_yield_rainfed_ana"

        result = pd.DataFrame(np.full([modelFrame['year'].unique().shape[0],3], np.nan), index=modelFrame['year'].unique(), columns=['R2','rmse','R2_classic'])

        for y in range(modelFrame['year'].min(), modelFrame['year'].max()+1):
            con = modelFrame['year']==y
            yieldType = "yield_rainfed_ana"
            r2_temp = modelFrame.loc[con,[yieldType, \
                                    model]].corr() \
                [model][0]**2
            
            # N is the sample number after removing nan
            N = modelFrame.loc[con,[yieldType,model]].dropna().shape[0]
            rmse_temp = (((modelFrame.loc[con, model] -  \
                              modelFrame.loc[con, yieldType])**2).sum() \
                                          /N)**0.5
                                      #    /modelFrame.loc[con,yieldType].shape[0])**0.5
                         
    #                                       /modelFrame.loc[con,model].shape[0])**0.5

            sst = ((modelFrame.loc[con, yieldType] \
                    - modelFrame.loc[con, yieldType].mean())**2).sum()
            sse = ((modelFrame.loc[con, yieldType] - modelFrame.loc[con, model])**2).sum()

            result.loc[y] = [r2_temp, rmse_temp, 1-sse/sst]
        print(model)
        # print(result)
        print(result)
        print(result.median()['rmse'])
        print(result.median()['R2'])

def calculateTotalYield():
    D = pd.read_csv("vpd_spline_evi_poly")
    D = D.loc[~pd.isnull(D["yield_rainfed"]),:]
    df = pd.DataFrame(index=[i for i in range(2003,2017,1)],columns=["Predicted National Rainfed Yield", "Actual National Rainfed Yield", "100 * (Predicted - Actual)/Actual"])
    df = df.fillna(0)
    for year in range(2003,2017,1):
        C = D.loc[D["year"] == year,:]
        predQuant = C["Predicted_yield_rainfed"]
        yieldQuant = C["yield_rainfed"]
        areaQuant = C["area_rainfed"]
        print(("{:,}".format(np.dot(yieldQuant, areaQuant)),"{:,}".format(np.dot(predQuant, areaQuant))))
        actualYield = np.dot(yieldQuant, areaQuant)
        predictedYield = np.dot(predQuant, areaQuant)
        actualYield_str = "{:,}".format(actualYield)
        predictedYield_str = "{:,}".format(predictedYield)
        df.loc[year,:] = predictedYield_str, actualYield_str, ((predictedYield/actualYield) - 1) * 100
    print(df)
    print(os.getcwd())
    df.to_csv("national_yield_stats.csv")

# def printRelativeYield(csvFiles):
    # os.chdir("../dataFiles")
    # D = pd.read_csv("soybean_handled_dataset")
    # yearRange = range(2003,2017,1)
    # for aFile in csvFiles:
        # for year in yearRange:
            # E = D[D["year"] == year]


if __name__ == "__main__":
    models = ['Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly',#4
              ]
    # calculateR2Emergency(models)
    # calculateTotalYield()

    os.chdir('../dataFiles/newR_Prediction_CSVs')
    # calculateR2Emergency(models)

    calculateTotalYield()


    # calculateR2(['vpd_spline_evi_poly_2'])
