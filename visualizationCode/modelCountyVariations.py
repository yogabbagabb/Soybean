import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sys
import os
import pdb
sys.path.append("./Crop_modeling-master")
from plot_model_performance_interannual_variability_causes import plot_name, plot_fitting
homePath = os.getcwd()


def prediction_result_by_county(test,yield_type='rainfed', minNumYears = 0):
    """
    Get R2, RMSE and R2_classic statistics for each county.
    :param test:
    :param yield_type:
    :param minNumYears:
    :return:
    """

    # Create a multi-index dataframe to save state results
    yield_type_dict = {'all': 'yield', 'rainfed':'yield_rainfed','irrigated':'yield_irr'}
    fipsCodes = test['FIPS'].unique()
    result = pd.DataFrame(np.full([fipsCodes.shape[0],3], np.nan), index=fipsCodes, columns=['R2','rmse','R2_classic'])

    for co in fipsCodes:
        con = (test['FIPS']==co)
        
        

        # N is the sample number after removing nan
        N = test.loc[con,[yield_type_dict[yield_type],'Predicted_'+yield_type_dict[yield_type]]].dropna().shape[0]
        if (N == 0) or (N < minNumYears):
            result = result.drop(co)
            continue




        r2_temp = test.loc[con,[yield_type_dict[yield_type],
                            'Predicted_'+yield_type_dict[yield_type]]].corr() \
        ['Predicted_'+yield_type_dict[yield_type]][0]**2
        
        rmse_temp = (((test.loc[con, 'Predicted_'+yield_type_dict[yield_type]] - 
                      test.loc[con, yield_type_dict[yield_type]])**2).sum() \
                                  /N)**0.5
                                 # /test.loc[con,yield_type_dict[yield_type]].shape[0])**0.5
                 
#                                       /test.loc[con,'Predicted_'+yield_type_dict[yield_type]].shape[0])**0.5

        sst = ((test.loc[con, yield_type_dict[yield_type]] 
                - test.loc[con, yield_type_dict[yield_type]].mean())**2).sum()

        sse = ((test.loc[con, yield_type_dict[yield_type]] - test.loc[con, 'Predicted_'+yield_type_dict[yield_type]])**2).sum()

        if (sst != 0):
            result.loc[co] = [r2_temp, rmse_temp, 1-sse/sst]

        else:
            result.loc[co] = [r2_temp, rmse_temp, np.nan]
    return result        

def getRainfedYieldSTD(stateFilter = [], minNumYears = 1):
    """
    Get the standard deviation of rainfed yield for the original soybean dataset.
    :param stateFilter:
    :param minNumYears:
    :return:
    """


    os.chdir("../dataFiles/")
    D = pd.read_csv("soybean_handled_dataset")
    if stateFilter != []:
        D = D.loc[D["State"] == stateFilter[0],:]
    os.chdir(homePath)

    fipsCodes = D['FIPS'].unique()
    result = pd.DataFrame(np.full([fipsCodes.shape[0],1], np.nan), index=fipsCodes, columns=['STD of yield_rainfed'])

    D = D.loc[D["year"] >= 2003, :]
    D = D.loc[~pd.isnull(D['yield_rainfed']),:]

    for co in fipsCodes:
        con = (D['FIPS']==co)
        if (np.sum(con) <= 1 or np.sum(con) <= minNumYears):
            result = result.drop(co)
            continue

        result.loc[co] = (D.loc[con,"yield_rainfed"]).std()

    return result



def plotInterannualVar(minNumYears, states =[], globalPlot=False):
    """
    Plot rainfed yield standard deviation by
    variation in R2 and RMSE  for vpd_spline_evi_poly, the best model. Plot this variation
    across all states and, then, for each state.
    :param minNumYears: The minimum number of years of data (entries) that a county within a state should have.
    :param states: The states to construct plots for.
    :param globalPlot: Look at the temporal yield variability across all states if true; otherwise, just use one state.
    :return:
    """
    os.chdir("../dataFiles/newR_Prediction_CSVs/")
    D = pd.read_csv("vpd_spline_evi_poly")
    os.chdir(homePath)

    if (globalPlot):

        P = prediction_result_by_county(D,minNumYears=minNumYears)
        S = getRainfedYieldSTD(minNumYears = minNumYears)
        # The order in what follows is important
        # P.join(S) is different from S.join(P) in that the former
        # merges on P's index values; the latter merges on S's index values
        frame = P.join(S,how="inner")
        print(frame)

        fig, axes = plt.subplots(1,2,figsize=(10,7.5))
        frame.plot.scatter(x='STD of yield_rainfed',y='R2',ax=axes[0])
        plot_name(frame, 'STD of yield_rainfed','R2',axes[0],fontsize=10)
        plot_fitting(frame, 'STD of yield_rainfed','R2',axes[0],order=1)

        frame.plot.scatter(x='STD of yield_rainfed',y='rmse',ax=axes[1])
        plot_name(frame, 'STD of yield_rainfed','rmse',axes[1],fontsize=10)
        plot_fitting(frame, 'STD of yield_rainfed','rmse',axes[1],order=1)

        axes[0].set_title('R2',fontsize=12)
        axes[1].set_title('RMSE (bu/ac)',fontsize=12)


        axes[0].set_xlabel('Temporal yield variability (bu/ac)',fontsize=12)
        axes[1].set_xlabel('Temporal yield variability (bu/ac)',fontsize=12)

        plt.suptitle("Temporal yield variability across all states -- %s Observations" %(str(minNumYears)))
        plt.savefig('./figures/interannualVariation/interannual_var_global_%s'%(str(minNumYears)))
        print('figure saved')

    else:
        for state in states:
            P = prediction_result_by_county(D,minNumYears=minNumYears)
            S = getRainfedYieldSTD([state],minNumYears=minNumYears)
            # The order in what follows is important
            # P.join(S) is different from S.join(P) in that the former
            # merges on P's index values; the latter merges on S's index values
            frame = S.join(P,how="inner")
            if frame.shape[0] == 0:
                continue

            fig, axes = plt.subplots(1,2,figsize=(10,7.5))
            frame.plot.scatter(x='STD of yield_rainfed',y='R2',ax=axes[0])
            plot_name(frame, 'STD of yield_rainfed','R2',axes[0],fontsize=10)
            plot_fitting(frame, 'STD of yield_rainfed','R2',axes[0],order=1)

            frame.plot.scatter(x='STD of yield_rainfed',y='rmse',ax=axes[1])
            plot_name(frame, 'STD of yield_rainfed','rmse',axes[1],fontsize=10)
            plot_fitting(frame, 'STD of yield_rainfed','rmse',axes[1],order=1)

            axes[0].set_title('R2',fontsize=12)
            axes[1].set_title('RMSE (bu/ac)',fontsize=12)


            axes[0].set_xlabel('Temporal yield variability (bu/ac)',fontsize=12)
            axes[1].set_xlabel('Temporal yield variability (bu/ac)',fontsize=12)

            plt.suptitle("Temporal yield variability in %s -- %s Observations" %(state,str(minNumYears)))
            plt.savefig('./figures/interannualVariation/interannual_var_%s_%s'%(state,str(minNumYears)))
            print('figure saved')
            # plt.show()

if __name__ == "__main__":
    os.chdir("../dataFiles/")
    D = pd.read_csv("soybean_handled_dataset")
    os.chdir(homePath)
    states = D["State"].unique()

    for state in states:
        plotInterannualVar(7,[state],globalPlot=False)
        plotInterannualVar(13,[state],globalPlot=False)

    plotInterannualVar(7,globalPlot=True)
    plotInterannualVar(13,globalPlot=True)



