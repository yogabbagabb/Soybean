import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def interannual():
    df = pd.read_csv('../dataFiles/newR_Prediction_CSVs/statsDirectory/vpd_spline_evi_poly_stats.csv')
    colorArray = ['b-']
    colorArray2 = ['bo']
    f,ax = plt.subplots(1,2)
    # colorArray = ['b-','g-','r-','y-','c-']
    # colorArray2 = ['bo','go','ro','yo','co']
    years = [2000 + i for i in range(3,17)]
    ax[0].plot(years, df['rmse'], colorArray[0])
    ax[0].plot(years, df['rmse'], colorArray2[0])
    ax[0].set_title("Interannual RMSE variation")
    ax[0].set_ylabel("RMSE")
    ax[0].set_xlabel("Years")
    ax[1].plot(years, df['R2'], colorArray[0])
    ax[1].plot(years, df['R2'], colorArray2[0])
    ax[1].set_title("Interannual R2 variation")
    ax[1].set_ylabel("")
    ax[1].set_xlabel("Years")

    fig = plt.gcf()
    fig.canvas.set_window_title('Interannual Variation')
    

    # plt.plot(years, avgYield, colorArray2[0])
    # plt.plot(years, 0.0628*df_rmse.loc[:,model],colorArray[i], label=model)

    plt.show()
    



# A: Global Regression (yield ~ X5 + X6 + X7 + X8 + Y5 + Y6 + Y7 + Y8)
# B: Varying Slope (X) + Varying Intercept + Fixed Slope (Y) + No Correlation in Slope and Intercept
# C: Varying Slope (X) + Varying Intercept + Varying Slope (Y) + No Correlation in Slope and Intercept
# D: Varying Slope (X) + Varying Intercept + Varying Slope (Y5 + Y6) + Fixed Slope (Y7 + Y8)  + No Correlation in Slope and Intercept
# E: Varying Slope (X) + Varying Intercept + Fixed Slope (Y5 + Y6) + Varying Slope (Y7 + Y8)  + No Correlation in Slope and Intercept



if __name__ == "__main__":
    # boxplot()
    interannual()


