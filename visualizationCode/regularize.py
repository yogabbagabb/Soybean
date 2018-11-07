import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from matplotlib import pyplot as plt
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn import preprocessing


import os 
import sys

homePath = os.getcwd()


def getData(predicType):
    os.chdir("../dataFiles")
    frame = pd.read_csv("soybean_handled_dataset")
    os.chdir(homePath)
    nonNAEntries = ~pd.isnull(frame[predicType + "5"])
    for j in range(6,10):
        nonNAEntries = np.logical_and(nonNAEntries, ~pd.isnull(frame[predicType + str(j)]))

    nonNAEntries = np.logical_and(nonNAEntries, ~pd.isnull(frame["yield_rainfed_ana"]))

    nonNAFrame = frame.loc[nonNAEntries, :]
    return nonNAFrame[[predicType + "5", predicType + "6", predicType + "7", predicType + "8", predicType + "9"]], nonNAFrame[["yield_rainfed_ana"]], nonNAFrame

def makeTestFold(frame):
    split = np.empty(frame.shape[0],dtype=int)
    before2003 = frame["year"] < 2003
    split[before2003] = int(-1)
    startingYear = 2003
    for i in range(0,14,1):
        year = startingYear + i
        indexesToChange = frame["year"] == year
        split[indexesToChange] = int(i)
    print(list(split))
    return PredefinedSplit(test_fold=split)




def makePlot(predicType):
    X,Y,frame = getData(predicType)
    X = preprocessing.scale(X)
    # print(X)
    # print(Y)
    theCVIterator = makeTestFold(frame)


    alphas = np.logspace(-3, 0.5, 50)
    mean_scorer = make_scorer(mean_squared_error)

    plt.figure(figsize=(5, 3))

    for Model in [Lasso, Ridge, ElasticNet]:
        scores = [cross_val_score(Model(alpha), X, Y, scoring=mean_scorer, cv=theCVIterator).mean() for alpha in alphas]
        plt.plot(alphas, scores, label=Model.__name__)

    plt.legend(loc='upper left')
    plt.xlabel('alpha')
    plt.ylabel('cross validation score')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    makePlot("precip")
