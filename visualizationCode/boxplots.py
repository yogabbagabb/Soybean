import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

def boxplot(rmse=True):
    """
    Create boxplots to view the spread of statistics (rmse and R2) of a single model from 2003 to 2016.
    The standard models are those defined below in the object called models.
    :param rmse: View the spread of RMSE (if false, view the spread of R2).
    :return: Nothing
    """

    length = 4
    width = 6
    f, axarr = plt.subplots(length, width)

    models = ['Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly',#4
              ]
    if rmse:
        f.tight_layout()
        for i,m in enumerate(models):
            A = pd.read_csv(m + "_stats.csv")
            axarr[int(i/width), int(i%width)].boxplot(A["rmse"])
            axarr[int(i/width), int(i%width)].set_title(str(m))
            axarr[int(i/width), int(i%width)].set_yticks(np.arange(3,9,1))
            axarr[int(i/width), int(i%width)].text(0,1.5,"Median: " + str(round(A["rmse"].median(),4)))
        f.canvas.set_window_title('soybean_RMSE')
            
        plt.show()

    else:
        f.tight_layout()
        for i,m in enumerate(models):
            A = pd.read_csv(m + "_stats.csv")
            axarr[int(i/width), int(i%width)].boxplot(A["R2"])
            axarr[int(i/width), int(i%width)].set_title(str(m))
            axarr[int(i/width), int(i%width)].text(0,min(A["R2"])-0.05,"Median: " + str(round(A["R2"].median(),4)))
        f.canvas.set_window_title('soybean_R2')
        plt.show()


def boxplotTogether(rmse=True):
    """
    Superimpose calls to boxplot (see above) on the same chart.
    :param rmse: Same as above:
    View the spread of RMSE (if false, view the spread of R2).
    :return: Nothing
    """
    if rmse:

        # The following object uses the following order of models deliberately. In this order,
        # median RMSEs are sorted.
        models = ['vpd_spline_evi_poly', 'vpd_poly_evi_poly', 'lstmax_spline_evi_poly',
       'vpd_spline_evi', 'tave_spline_evi', 'tave_spline_evi_poly',
       'vpd_spline', 'evi_spline_only', 'evi_poly_only', 'vpd_poly',
       'evi_linear_only', 'lstmax_poly_evi_poly_only',
       'lstmax_spline_evi_poly_only', 'tave_spline', 'tave_poly', 'vpd_linear',
       'tave_linear', 'Tgs_poly', 'lstmax_poly_only', 'Tgs_linear',
       'lstmax_spline_only', 'lstmax_linear_only']
        csvBox = []
        for i,m in enumerate(models):
            A = pd.read_csv(m + "_stats.csv")
            csvBox.append(A['rmse'])
        plt.boxplot(csvBox)
        plt.xticks([i for i in range(1,23,1)], models, rotation=67.5)
        plt.ylabel("RMSE (Bushels/Acre)")
        f = plt.gcf()
        f.tight_layout()
        f.canvas.set_window_title('Soybean RMSE Series')
        ax = plt.gca()
        for i, xpos in enumerate(ax.get_xticks()):
            ax.text(xpos, -1, "Median: %s"%(round(csvBox[i].median(),4)),rotation=67.5)
        plt.show()

    else:
        # The following object uses the following order of models deliberately. In this order,
        # median R2s are sorted.
        models = ['Tgs_linear', 'Tgs_poly', 'tave_linear', 'lstmax_poly_only',
       'lstmax_spline_only', 'lstmax_linear_only', 'vpd_linear',
       'evi_linear_only', 'tave_poly', 'vpd_spline', 'vpd_poly', 'tave_spline',
       'lstmax_spline_evi_poly_only', 'evi_poly_only',
       'lstmax_poly_evi_poly_only', 'evi_spline_only',
       'lstmax_spline_evi_poly', 'vpd_spline_evi_poly', 'tave_spline_evi_poly',
       'vpd_poly_evi_poly', 'vpd_spline_evi', 'tave_spline_evi']
        csvBox = []
        for i,m in enumerate(models):
            A = pd.read_csv(m + "_stats.csv")
            csvBox.append(A['R2'])
        plt.boxplot(csvBox)
        plt.xticks([i for i in range(1,23,1)], models, rotation=67.5)
        plt.ylabel("R2")
        f = plt.gcf()
        f.tight_layout()
        f.canvas.set_window_title('Soybean R2 Series')
        ax = plt.gca()
        for i, xpos in enumerate(ax.get_xticks()):
            ax.text(xpos, -1, "Median: %s"%(round(csvBox[i].median(),4)),rotation=67.5)
            print(csvBox[i])
        plt.show()

def sortMedians():
    """
    Print two lists of model names that are sorted where the sorting order is the model's median R2 and RMSE
    :return:
    """
    models = ['Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly'
              ]
    medianBoxRMSE = []
    medianBoxR2 = []
    for i,m in enumerate(models):
        A = pd.read_csv(m + "_stats.csv")
        medianBoxRMSE.append(A['rmse'].median())
        medianBoxR2.append(A['R2'].median())
    
    rmseSeries = pd.Series(data=medianBoxRMSE, index=models)
    R2Series = pd.Series(data=medianBoxR2, index=models)
    rmse_sorted = rmseSeries.sort_values()
    r2_sorted = R2Series.sort_values()
    print(rmse_sorted.index)
    print(r2_sorted.index)


if __name__ == "__main__":
    # boxplot()
    # boxplot(False)
    os.chdir("../dataFiles/newR_Prediction_CSVs/statsDirectory/")
    boxplotTogether()
    boxplotTogether(False)
    # sortMedians()

    
