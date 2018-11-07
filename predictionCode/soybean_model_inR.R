# This file calculates predicted_yield_rainfed without detrending
# It does not calcualte predicted_yield_irr or predicted_yield

# It calculates RMSE
# It does not calculate R^2

# In[1]:

rm(list=ls())
require(lme4)
require(nlme)
require(splines)
require(magrittr)
year = seq(from = 1981, to = 2016);pred.year = seq(from=23, to=36)


RMSE = function(m, o){
sqrt(mean((m - o)^2))
}


# In[14]:

newRMSE = function(m, o,length){
sqrt(sum((m - o)^2)/length)
}



calculateRMSE = function(yield_prediction_csv,rmse_csv)
{
    # In[7]:

    results <- read.csv(yield_prediction_csv)

    firstModelIndex <- 8
    model_names <- colnames(results[firstModelIndex:length(colnames(results))])
    lastModelIndex <- firstModelIndex + length(model_names) - 1


    numberYears <- 14
    model_rmse_mat <- matrix(, nrow = numberYears, ncol = length(model_names))

    lastModelIndex <- firstModelIndex + length(model_names) - 1

    for (i in 1:numberYears){
        print(i)
        test.year = year[pred.year[i]]
        #train.year = year[-pred.year[i]]
        yearIndex <- 2
        data.test = results[which(results[, yearIndex] %in% test.year),]
        for (j in firstModelIndex:lastModelIndex)
        {
            indexInModelNames <- j - firstModelIndex + 1
            model_string <- (model_names[indexInModelNames])
            data.test.copy = data.test[!is.na(data.test[,model_string]),]
            data.test.copy = data.test.copy[!is.na(data.test.copy[,"yield_rainfed"]),]
            model_rmse_mat[i,indexInModelNames] <- RMSE(data.test.copy$"yield_rainfed", data.test.copy[,model_string])
        }
    }

    df <- data.frame(model_rmse_mat)
    colnames(df) <- model_names
    write.csv(df, rmse_csv)
    print(df)
    print(apply(df,2,median))
    print(apply(df,2,mean))
}






initialize = function()
{
    # In[3]:

    mydata <- read.csv("../dataFiles/soybean_handled_dataset")
    #ind <- sapply(mydata, is.numeric)
    #ind["year"] <- FALSE
    #ind["State"] <- FALSE
    #ind["FIPS"] <- FALSE
    #ind["yield_rainfed"] <- FALSE
    #ind["yield_rainfed_ana"] <- FALSE
    #ind["yield"] <- FALSE
    #ind["soy_percent"] <- FALSE
    #mydata[ind] <- scale(mydata[ind])
    

    IA <- 19;IL <- 17;IN <- 18;MN <- 27;OH <- 39;WI <- 55;MO <- 29;NB <- 31;KS <- 20;SD <- 46
    three_states = c(IL,IN,IA)
    seven_states <- c(IA, IL, IN, MN, OH, WI, MO)
    ten_states <- c("IOWA", "ILLINOIS", "INDIANA", "MINNESOTA", "OHIO", "WISCONSIN", "MISSOURI", "NEBRASKA", "KANSAS", "SOUTH DAKOTA")
    twelve_states <- c("IOWA", "ILLINOIS", "INDIANA", "MINNESOTA", "OHIO", "WISCONSIN", "MISSOURI", "NEBRASKA", "KANSAS", "SOUTH DAKOTA","NORTH DAKOTA", "MICHIGAN")



    mydata$FIPS = as.factor(mydata$FIPS)
    mydata$State = as.factor(mydata$State)



    # In[5]:

    train_region <- twelve_states
    test_region <- twelve_states


    yield_prediction_csv_1 <- "./all_yan_models_prediction"
    rmse_csv_1 <- "./all_yan_models_rmse"

    model_formulas_1 <- c("yield_rainfed_ana ~ tave56789 + precip56789+ FIPS",

"yield_rainfed_ana ~ tave56789 + I(tave56789^2) + precip56789 + I(precip56789^2)+ FIPS",

"yield_rainfed_ana ~ tave6 + tave7 + tave8 + precip6 + precip7 + precip8 + precip9+ FIPS",

"yield_rainfed_ana ~ vpdave6 + vpdave7 + vpdave8 + precip6 + precip7 + precip8 + precip9+ FIPS",

"yield_rainfed_ana ~ tave6 + tave7 + tave8+ I(tave6^2) + I(tave7^2) + I(tave8^2) + precip6 + precip7 + precip8 + precip9+ I(precip6^2) + I(precip7^2) + I(precip8^2) + I(precip9^2)+ FIPS",

"yield_rainfed_ana ~ vpdave6 + vpdave7 + vpdave8 + I(vpdave6^2) + I(vpdave7^2) + I(vpdave8^2) + precip6 + precip7 + precip8 + precip9+ I(precip6^2) + I(precip7^2) + I(precip8^2) + I(precip9^2)+ FIPS",

"yield_rainfed_ana ~ lstmax6 + lstmax7 + lstmax8+ FIPS",

"yield_rainfed_ana ~ lstmax6 + lstmax7 + lstmax8+ I(lstmax6^2) + I(lstmax7^2) + I(lstmax8^2)+ FIPS",

"yield_rainfed_ana ~ evi5 + evi6 + evi7 + evi8 + evi9+ FIPS",

"yield_rainfed_ana ~ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ lstmax6 + lstmax7 + lstmax8+ I(lstmax6^2) + I(lstmax7^2) + I(lstmax8^2)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ vpdave6 + vpdave7 + vpdave8 + I(vpdave6^2) + I(vpdave7^2) + I(vpdave8^2) + precip6 + precip7 + precip8 + precip9+ I(precip6^2) + I(precip7^2) + I(precip8^2) + I(precip9^2)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(evi5,  degree=1, knots = c(0.20,0.25,0.33),df=4) + bs(evi6,  degree=1, knots = c(0.25,0.35),df=3) + bs(evi7,  degree=1, knots = c(0.22,0.61),df=3) + bs(evi8,  degree=1, knots = c(0.58,0.63),df=3) + bs(evi9, degree=1, knots = c(0.35,0.51),df=3)+ FIPS",

"yield_rainfed_ana ~ bs(lstmax6,  degree=1, knots = c(28.6, 33.1, 34.5),df=4)+ bs(lstmax7,  degree=1, knots = c(26.4,34.5),df=3)+ bs(lstmax8,  degree=1, knots = c(26.1,28.1,29.3),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(lstmax6,  degree=1, knots = c(28.6, 33.1, 34.5),df=4)+ bs(lstmax7,  degree=1, knots = c(26.4,34.5),df=3)+ bs(lstmax8,  degree=1, knots = c(26.1,28.1,29.3),df=4) + bs(precip6,  degree=1, knots = c(92.6,182.2),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,61,104.1),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(tave6,  degree=1, knots = c(20.6,22.6),df=3)+ bs(tave7,  degree=1, knots = c(22.5,25.5),df=3)+ bs(tave8,  degree=1, knots = c(21.5,23.5),df=3) + bs(precip6,  degree=1, knots = c(92.6,182.2),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,61,104.1),df=4)+ FIPS",

"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.552),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ FIPS",

"yield_rainfed_ana ~ bs(lstmax6,  degree=1, knots = c(28.780, 33.876),df=3)+ bs(lstmax7,  degree=1, knots = c(26.709,35.622),df=3)+ bs(lstmax8,  degree=1, knots = c(25.901,29.033),df=3)+ FIPS",

"yield_rainfed_ana ~ bs(tave6,  degree=1, knots = c(21.014, 23.182),df=3)+ bs(tave7,  degree=1, knots = c(22.537,25.448),df=3)+ bs(tave8,  degree=1, knots = c(21.55,24.815),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9+ FIPS",

"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5,14.96), df=5)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.552),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9+ FIPS",

"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,16.5521),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.096),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

#"yield_rainfed_ana ~ bs(vpdave6,  degree=1, knots = c(8.5,10.5,12.5),df=4)+ bs(vpdave7,  degree=1, knots = c(8,10.5),df=3) + bs(vpdave8,  degree=1, knots = c(8.06,9.10),df=3) + bs(precip6,  degree=1, knots = c(92.6,182.2),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428),df=3) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,61,104.1),df=4)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS",

"yield_rainfed_ana ~ bs(tave6,  degree=1, knots = c(21.014,23.182),df=3)+ bs(tave7,  degree=1, knots = c(22.537,25.448),df=3)+ bs(tave8,  degree=1, knots = c(21.55,24.815),df=3) + bs(precip6,  degree=1, knots = c(92.6,209.06),df=3) + bs(precip7,  degree=1, knots = c(56.191,89.1428,241.191),df=4) + bs(precip8,  degree=1, knots = c(50.22, 75.40),df=3) + bs(precip9,  degree=1, knots = c(29.9037,104.1),df=3)+ evi5 + evi6 + evi7 + evi8 + evi9 + I(evi5^2) + I(evi6^2) + I(evi7^2) + I(evi8^2) + I(evi9^2)+ FIPS")




    model_names_1 <- c('Tgs_linear','Tgs_poly','tave_linear','vpd_linear','tave_poly','vpd_poly', #6
              'lstmax_linear_only','lstmax_poly_only','evi_linear_only','evi_poly_only', #4
              'lstmax_poly_evi_poly_only','vpd_poly_evi_poly', #2
              'evi_spline_only', 'lstmax_spline_evi_poly_only',#2
              'lstmax_spline_evi_poly','tave_spline','vpd_spline','lstmax_spline_only', #4
              'tave_spline_evi', 'vpd_spline_evi', 'vpd_spline_evi_poly','tave_spline_evi_poly')#4 
    
    fitting_functions_1 <- c("lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm","lm")
    svd_issue_1 <- c("N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N","N")
    uses_FIPS_1 <- c("Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y")
    uses_evi_1 <- c("N","N","N","N","N","N","N","N","Y","Y","Y","Y","Y","Y","Y","N","N","N","Y","Y","Y","Y")
    uses_lstmax_1 <- c("N","N","N","N","N","N","Y","Y","N","N","Y","N","N","Y","Y","N","N","Y","N","N","N","N")


    yield_prediction_csv_list <- list(yield_prediction_csv_1)
    rmse_csv_list <- list(rmse_csv_1)
    model_formulas_list <- list(model_formulas_1)
    model_names_list <- list(model_names_1)
    fitting_functions_list <- list(fitting_functions_1)
    svd_issue_list <- list(svd_issue_1)
    uses_evi_list <- list(uses_evi_1)
    uses_lstmax_list <- list(uses_lstmax_1)
    uses_FIPS_list <- list(uses_FIPS_1)



    number_models <- length(yield_prediction_csv_list)


    for (model_index in (1:number_models))
    {
        yield_prediction_csv <- yield_prediction_csv_list[[model_index]]
        rmse_csv <- rmse_csv_list[[model_index]]
        model_formulas <- model_formulas_list[[model_index]]
        model_names <- model_names_list[[model_index]]
        fitting_functions <- fitting_functions_list[[model_index]]
        svd_issues <- svd_issue_list[[model_index]]
        uses_evi <- uses_evi_list[[model_index]]
        uses_lstmax <- uses_lstmax_list[[model_index]]
        uses_FIPS <- uses_FIPS_list[[model_index]]

        predictions <- vector("list", length(model_formulas))

        if (length(model_formulas) != length(model_names))
        {
            stop("The length of model_formulas != length of model_names")
        }

        for (j in 1:length(model_formulas))
        {
            startTime <- Sys.time()
            print(model_names[j])
            endYear <- 2016
            startYear <- 1981
            for (i in 1:14){

                if (i == 1)
                {
                    v <- c(1981:2002, 2004:endYear)
                }

                else if (i == 14)
                {
                    v <- startYear:year[pred.year[i-1]]
                }

                else
                {
                    v <- c(startYear:year[pred.year[i-1]], year[pred.year[i+1]]:endYear)
                }


                print(i)
                print(v)
                train.year <- v
                test.year = year[pred.year[i]]
                #train.year = year[-pred.year[i]]

                #data.train = mydata[which(mydata[, 1] %in% train.year & mydata$State %in% train_region),]
                data.train = mydata[!(mydata$"year" %in% year[pred.year[i]]),]
                print(dim(data.train))


                data.test = mydata[mydata$"year" %in% test.year,]
                                         #& mydata$State %in% test_region),]

                soybean_percent_min <- 0

                if (svd_issues[j] == "Y")
                {
                    soybean_percent_min <- 0.001
                }

                #print(colnames(data.train))
                #print(data.train$'soybean_percent')
                data.train <- data.train[data.train$'soybean_percent' > soybean_percent_min,]
                print(dim(data.train))
                data.test <- data.test[data.test$'soybean_percent' > soybean_percent_min,]

                if (uses_evi[j] == "Y")
                {
                    data.train <- data.train[!is.na(data.train$"evi5"),]
                    data.test <- data.test[!is.na(data.test$"evi5"),]
                }

                if (uses_lstmax[j] == "Y")
                {
                    data.train <- data.train[!is.na(data.train$"lstmax5"),]
                    data.test <- data.test[!is.na(data.test$"lstmax5"),]
                }
                # tave5 is null iff precip5 is null iff tave5 is null
                data.train <- data.train[!is.na(data.train$'tave5'),]
                data.test <- data.test[!is.na(data.test$'tave5'),]



                model_function <- NULL

                if (fitting_functions[j] == "lm")
                {
                    model_function <- lm(as.formula(model_formulas[j]), data=data.train)

                }
                else if (fitting_functions[j] == "lmer")
                {
                    model_function <- lmer(as.formula(model_formulas[j]), data=data.train, control = lmerControl(optimizer ="Nelder_Mead"))
                }


                if (uses_FIPS[j] == "Y")
                {
                    non_na_rainfed_entries <- data.train[!(is.na(data.train$'yield_rainfed')),]
                    data.test <- data.test[data.test$"FIPS" %in% non_na_rainfed_entries$"FIPS",]
                    #data.test <- remove_missing_levels(model_function, data.test)
                }

                #data.test <- data.test[data.test$"FIPS" %in% data.train$"FIPS",]


                if (i == 14)
                {
                    s <- capture.output(summary(model_function))
                    coeff <- capture.output(coef(model_function))
                    coeff_mean <- capture.output(coef(summary(model_function)))
                    csv_name <- paste0(yield_prediction_csv)
                    csv_name <- substr(csv_name, 3, nchar(csv_name)) 
                    #csv_name <- substr(csv_name, 1, nchar(csv_name) - 4)
                    print(csv_name)
                    write(s, paste0('./coefficients_', csv_name, model_names[j],'.txt'), append=TRUE)
                    write("\n_________________\n",paste0('./coefficients_', csv_name, model_names[j],'.txt'), append=TRUE)
                    write(coeff, paste0('./coefficients_', csv_name, model_names[j],'.txt'), append=TRUE)
                    write("\n_________________\n",paste0('./coefficients_', csv_name, model_names[j],'.txt'), append=TRUE)
                    write(coeff_mean, paste0('./coefficients_', csv_name, model_names[j],'.txt'), append=TRUE)
                    theFile <-  paste0(model_names[j], "%03d.png")
                    png(filename=theFile)
                    plot(model_function,ask=FALSE)
                    dev.off()
                }







                #assign(paste0("data.test",test.year), mydata[which(mydata[, 1] %in% test.year & mydata$State %in% test_region),])
                assign(paste0("data.test",test.year), data.test)
                assign(paste0("func.pred",test.year), rep(NA,dim(get(paste0("data.test",test.year)))[1]))
                #print(all(get(paste0("data.test",test.year))$"FIPS" %in% data.train$"FIPS"))

                assign(paste0("func.pred",test.year), predict(model_function,get(paste0("data.test",test.year)))) #+ yearly.means[which(mydata[, 1] %in% test.year & mydata$State %in% test_region)])   


            }

            #Yan <- rbind.data.frame(data.test2016)

            #predictions[[j]] <- c(func.pred2016)

            Yan <- rbind.data.frame(data.test2003,data.test2004,data.test2005,data.test2006,data.test2007,data.test2008, data.test2009, data.test2010, data.test2011, data.test2012, data.test2013, data.test2014, data.test2015, data.test2016)

            predictions[[j]] <- c(func.pred2003,func.pred2004,func.pred2005,func.pred2006,func.pred2007,func.pred2008, func.pred2009, func.pred2010, func.pred2011, func.pred2012, func.pred2013, func.pred2014, func.pred2015, func.pred2016)
            temp <- data.frame(predictions[[j]])
            write.csv(data.frame(temp, Yan), model_names[[j]])
            endTime <- Sys.time()
            print(endTime - startTime)


        }


        df <- data.frame(predictions)
        colnames(df) <- model_names
        Yan.data <- data.frame(Yan, df)



        # In[6]:

        basic_data <- c("year","State","FIPS","yield_ana", "yield_rainfed", "area_rainfed")
        desired_data <- c(basic_data, model_names)
        yield_prediction <- subset(Yan.data, select = desired_data)
        #desired_data[[length(desired_data)]] = "Predicted_yield_rainfed"


        # Add this back in to allow subsequent use of Yan's function to calculate RMSEs
        #colnames(yield_prediction)[ncol(yield_prediction)] <- "Predicted_yield_rainfed"

        trend_rainfed <- lm(yield_rainfed ~ year, data=mydata) # Use linear trends
        yearly.means <- predict(trend_rainfed, yield_prediction)
        for (i in 1:length(model_names))
        {
            yield_prediction[,model_names[[i]]]  <- yield_prediction[,model_names[[i]]] + yearly.means
        }


        write.csv(yield_prediction, yield_prediction_csv)
        for (j in 1:length(model_names))
        {
            file.remove(model_names[[j]])
        }
        calculateRMSE(yield_prediction_csv, rmse_csv)





    }
}

# Main method

options(warn=1)
initialize()


