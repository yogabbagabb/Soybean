#Code Overview

##dataFiles
###R_Prediction_CSVs/
	Contains the data frames that are formed after making predictions and concatenating
	the predictions (as a column) to the original dataset that the predictions' model
	was bsed upon

##yanModelStats/
	Contains statistics for RMSE and R2 than one using LOO Cross
	Validation on each of the years from 2003 to 2016.

##soybean_handled_dataset (csv file)
	The dataset obtained by preprocessing the starting CSV, see below.

##soybean_model_data_2017.csv
	The starting csv.




##flotsam
_______
###addToGit.py
	Add all the prediction csv files that are obtained through R prediction code using
	git add.



___________
##predictionCode

###print_model_comparisons.py
	Functions that Yan wrote to obtain model statistics; he wrote these to compare model
	statistics with the ones Aahan was getting. This was done so that Aahan could verify that
	his model was consistent with the original ones.

###save_dataset.py
	This is an adaptation of Yan's pre-processing code to obtain the pre-processed dataset
	called soybean_handled_dataset.

###soybean_model_inR.R
	Code in R for performing predictions. This is more robust that soybean_new_model.
	IMPORTANT:
	    This function will ultimately produce a prediction data frame and an RMSE data frame.
	    This data frame contains the predicted yields of all models ran in an execution of
	    soybean_model_inR.R and the RMSEs of these models. The names of these CSVS are, respectively,
	    as 11/7/18:
	        "./all_yan_models_prediction" and "./all_yan_models_rmse"

	    Side Effects:
	        This code produces diagnostics of several of each model on the year 2016. If you wish
	        to disable this, please comment out the code doing this (at the end of the file).

	        It also temporarily obtains a prediction data frame for each model and saves it locally. These
	        CSVS are removed in the event that all models within a batch run to completion. A batch is
	        just a collection of models run during one execution.

	        These diagnostics and files will appear in the predictionCode file.


###soybean_new_model.py
	An adaptation of Yan's prediction model for corn to soybean

###catalog.R
	Stores the model configurations that can be loaded into soybean_model_inR.R



____________
##visualizationCode

###examineData
	Examine whether we can say that
		Given a type of predictor X, the variants of the predictor X5 through X9
		are each not null iff only the remaining predictors are not null
	Aside (not useful for anyone but Aahan)
	All Prediction CSVs produced in R have
	their yield anomaly predictions saved
	as "predictions..j..". This function renames the column to "Predicted_yield_rainfed" before finally adding the trend function's trend value to obtain "Predicted_yield_rainfed".


###printPlots
		Prints response plots (linear,
		quadratic cubic) for certain,
		specifiable predictors.

###getKnots
	Print loess regression functions to identify knots for several types of predictors

###showKnots (this is better than getKnots)
	Print loess regression functions to identify knots for several types of predictors

###.printStats
	This is a hidden file (it is precede by . since it was designed for Aahan's convenience)
	Determine the R2 and RMSE using a prediction csv:
	calculateR2
		For prediction csvs obtained using Aahan's function if Aahan's function
		succeeds
	calculateR2Emergency
		For prediction csvs obtained using Aahan's function if Aahan's function
		fails to save all tested models in one csv file
	calculateR2EmergencyYan
		For prediction csvs obtained using Yan's function
	calculateTotalYield
		Calculates, for each year from 2003 to 2016, the actual national soybean
		yield and the predicted national soybean yield and the error between these
		two figures. These three statistics, for each year, are saved into a csv.
	printRelativeYield
		This will print the relative yield RMSE for each year for a given model configuration -- this is defined to be RMSE/(maxTrueYield - minTrueYield) and the median of such yields. If you pass in an array of model configurations, then you will obtain the foregoing statistics for each passed in model.

###printStrings.py
	Print, for each of Yan's models, the Python model code. "Yan's models" refer to the 22
	models he tested with corn, or that were tested in a forthcoming publication by him.

###boxplots.py
	Boxplots that list R2 and RMSE for all models

###interannual.py
	Interannual yield plot

###plotScatter.py
	Make scatter plots for yield versus predicted yield for vpd_evi_spline_poly.

###exploreND.py
	Examines the distribution of Temperature and VPD in North Dakota from 2003 to 2016
		The plots created from this code show that 2004 was a remarkably "cold" summer,
		which is consistent with this article:
			https://www.ndsu.edu/ndscoblog/?p=1072

###modelCountyVariations.py
	Plots county temporal rainfed yield variability against R^2 and RMSE statistics that
	are obtained using the vpd_spline_evi_poly model on soybean data.
	

###regularize.py
	In Development: A script that will determine which months are the most significant ones
	for precipitation and other predictors.


###Crop_modeling-master/plot_yield_response_lowess.py
	Plots each variable against yield anomaly and then displays the plot along with a loess plot.


