import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from scipy.stats import pearsonr



homePath = os.getcwd()


# mng = plt.get_current_fig_manager()
# print(mng.__dict__)
# mng.window.showMaximized()

def printCorrelation(monthNum):
    os.chdir("../dataFiles")
    D = pd.read_csv("soybean_handled_dataset")
    os.chdir(homePath)
    D = D.loc[~pd.isnull(D["tave5"]),:]
    D = D.loc[~pd.isnull(D["vpdave5"]),:]

    E = D[D["State"] == "NORTH DAKOTA"]
    corrFrame = pd.DataFrame(np.full((2,2),np.nan),index=['Global', 'North Dakota'], columns=['Pearson', 'Spearman'])

    corr = pearsonr(D['tave' + str(monthNum)], D['vpdave' + str(monthNum)])
    corrFrame.at['Global','Pearson'] =  corr[0]

    corr, _ = spearmanr(D['tave' + str(monthNum)], D['vpdave' + str(monthNum)])
    corrFrame.at['Global','Spearman'] = corr
    
    corr = pearsonr(E['tave' + str(monthNum)], E['vpdave' + str(monthNum)])
    corrFrame.at['North Dakota','Pearson'] = corr[0]
    
    corr, _ = spearmanr(E['tave' + str(monthNum)], E['vpdave' + str(monthNum)])
    corrFrame.at['North Dakota','Spearman'] = corr

    print(corrFrame)

def showDistribution():

    os.chdir("../dataFiles")
    D = pd.read_csv("soybean_handled_dataset")
    os.chdir(homePath)
    D = D[D["State"] == "NORTH DAKOTA"]
    yearRange = [i for i in range(2003,2017)]
    medianFrame = pd.DataFrame(np.full((len(yearRange),5),np.nan), index = yearRange, columns = ['vpd5','vpd6','vpd7','vpd8','vpd9']) 
    print(medianFrame)
    plt.figure(figsize=(20,10))
    plt.xlabel("Years")
    plt.ylabel("vpd")
    plt.title("vpd Trends in North Dakota from 2003-2017")
    for year in yearRange:
        B = D[D["year"] == year]
        yearBox = [year for i in range(0,B["vpd5"].shape[0])]
        plt.plot(yearBox, B["vpd5"], 'rx')
        medianFrame.at[year,"vpd5"] = B["vpd5"].median()

        yearBox = [year for i in range(0,B["vpd6"].shape[0])]
        plt.plot(yearBox, B["vpd6"], 'bx')
        medianFrame.at[year,"vpd6"] = B["vpd6"].median()

        yearBox = [year for i in range(0,B["vpd7"].shape[0])]
        plt.plot(yearBox, B["vpd7"], 'gx')
        medianFrame.at[year,"vpd7"] = B["vpd7"].median()

        yearBox = [year for i in range(0,B["vpd8"].shape[0])]
        plt.plot(yearBox, B["vpd8"], 'yx')
        medianFrame.at[year,"vpd8"] = B["vpd8"].median()

        yearBox = [year for i in range(0,B["vpd9"].shape[0])]
        plt.plot(yearBox, B["vpd9"], 'cx')
        medianFrame.at[year,"vpd9"] = B["vpd9"].median()
    os.chdir("../visualizationCode/figures/northDakotaAnomaly/")
    red_patch = mpatches.Patch(color='red', label='May')
    blue_patch = mpatches.Patch(color='blue', label='June')
    green_patch = mpatches.Patch(color='green', label='July')
    yellow_patch = mpatches.Patch(color='yellow', label='August')
    cyan_patch = mpatches.Patch(color='cyan', label='September')
    plt.legend(handles=[red_patch, blue_patch, green_patch, yellow_patch, cyan_patch])
    # plt.show()
    plt.savefig('vpd_' + 'points.png')
    os.chdir(homePath)
    
    print(medianFrame)
    plt.plot(yearRange, medianFrame.loc[:,"vpd5"], 'r-')
    plt.plot(yearRange, medianFrame.loc[:,"vpd6"], 'b-')
    plt.plot(yearRange, medianFrame.loc[:,"vpd7"], 'g-')
    plt.plot(yearRange, medianFrame.loc[:,"vpd8"], 'y-')
    plt.plot(yearRange, medianFrame.loc[:,"vpd9"], 'c-')
    os.chdir("../visualizationCode/figures/northDakotaAnomaly/")
    # plt.show()
    plt.savefig('vpd_' + 'pointsMedians.png')
    os.chdir(homePath)

    ax = plt.gca()
    months = [j for j in range(5,10,1)]
    colors = ['r','b','g','y','c']
    for num,col in zip(months,colors):
        ax.axhline(min(medianFrame.loc[:,"vpd"+str(num)]), alpha=0.250, c=col)




    # plt.legend(loc="best")
    os.chdir("../visualizationCode/figures/northDakotaAnomaly/")
    # plt.show()
    plt.savefig('vpd_' + 'pointsMediansMins.png')
    os.chdir(homePath)


if __name__ == "__main__":
    for i in [j for j in range(5,10,1)]:
        printCorrelation(i)

