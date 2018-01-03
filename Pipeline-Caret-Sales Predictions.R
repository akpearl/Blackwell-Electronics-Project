# Project name: predicted sales Volumes
# File name:    Existing Product Attributes (75% - train and 15% -test ds) 
#               New Product Attributes (predict ds)      
###############
# Housekeeping
###############

rm(list = ls()) # Clear objects if necessary
getwd()         # get working directory
                # set working directory 
setwd("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/R/Sales")

################
# Load packages
################
install.packages("e1071")  #needed for svm
install.packages("caret")
install.packages("corrplot")
##install.packages("doMC")  #package 'doMC' is not available for Windows
##install.packages("doSMP") #package 'doSMP' is not available for windows
#install.packages("doSNOW") # can be used for rf and glm
install.packages("randomForest") 

### install.packages("mnormt")
### install.packages("DEoptimR")

require(e1071)
require(randomForest)
require(caret)  #'ggplot2' -- object is masked from 'package:randomForest':margin
require(corrplot)
#require(doSNOW)
#require(mlbench)
#detectCores() ### used with doMC
#registerDoMC(cores = 4) ###  used with doMC
#require(mnormt)
#require(DEoptimR)
##############
# Import data
##############

# read train file; include header, preserve column names, use comma as column separator,
###               drop quotes, suppress factor conversion everywhere
exstprods <- read.csv("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/Weka Datasets/New Product Analysis Data/existing product attributes.csv", 
            header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)

newprods <- read.csv("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/Weka Datasets/New Product Analysis Data/new product attributes.csv", 
            header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)
# view data structure                
str(exstprods)  # 80 obs of 18 vars 
str(newprods)   # 17 obs of 18 vars

write.csv(newprods, "newprods.csv") # save copy of original ds in working directory

### rename columns
colnames(exstprods) <- c("ProductType","ProductNum", "Price","FiveStar","FourStar",
                         "ThreeStar","TwoStar","OneStar","PosRvw","NegRvw","WouldRecmnd",
                         "BestSell","ShipWeight","ProdDepth","ProdWidth","ProdHeight",
                         "ProfMarg","Volume")

colnames(newprods) <- c("ProductType","ProductNum", "Price","FiveStar","FourStar",
                         "ThreeStar","TwoStar","OneStar","PosRvw","NegRvw","WouldRecmnd",
                         "BestSell","ShipWeight","ProdDepth","ProdWidth","ProdHeight",
                         "ProfMarg","Volume")
################
# Evaluate data
################ 

str(exstprods) 
str(newprods) 

head(exstprods,5)
head(newprods,5)

tail(exstprods,5)
tail(newprods,5)

summary(exstprods) ## stats on each var (min, median, max, ect.)
summary(newprods)

#############
# Pre-process
#############

# remove obvious attribute
exstprods$ProductNum <- NULL
exstprods$ProfMarg <- NULL

# remove attributes from predict ds to match train ds
newprods$ProductNum <- NULL
newprods$ProfMarg <- NULL

# change data types
exstprods$BestSell <- as.numeric(exstprods$BestSell)
###Warning message:NAs introduced by coercion -- this means there are missing/non numeric values in this column
#see if and how many missing values are in BestSell column
sum(is.na(exstprods["BestSell"]))    # 15 NAs 

# populate missing values (NAs) in BestSell with mean
exstprods$BestSell[is.na(exstprods$BestSell)] <- mean(exstprods$BestSell, na.rm = TRUE)

# match format to train data set
newprods$BestSell <- as.numeric(newprods$BestSell)
# there were no missing values in the new DS in BestSell attribute

any(is.na(exstprods))  # confirm if any "NA" values in ds
any(is.na(newprods))   # confirm if any "NA" values in ds

# change data types to factor (currently discrete/int data type) to numeric
exstprods$Volume <- as.numeric(exstprods$Volume)
exstprods$ProductType <- factor(exstprods$ProductType)
#exstprods$ProductNum <- factor(exstprods$ProductNum)#no need to change will remove it anyway

# change data types to match train and predict ds
newprods$Volume <- as.numeric(newprods$Volume)
###Warning message:NAs introduced by coercion -- this means there are missing values, we need to predict them
newprods$ProductType <- factor(newprods$ProductType)

exstprods16v <- exstprods
newprods16v <- newprods

str(exstprods16v) #confirm changes # 80 obs of 16 vars
str(newprods16v)  #confirm changes # 17 obs of 16 vars

####################
# Feature selection
####################

# check for collinearity among num vars; do not include factors
correlations16v <- cor(exstprods16v[,c(2,3,4,5,6,7,8,9,10,12,13,14,15,16)])  
 
# print correlation table
print(correlations16v)   
 
# create correlation plot
corrplot(correlations16v, method = "circle") 

# check correlations and remove any features, like those with correlations >0.85 between predictors or 
## predictors and predicting var (Volume)
exstprods$FiveStar <- NULL
exstprods$OneStar <- NULL
exstprods$ThreeStar <- NULL
exstprods$NegRvw <- NULL

str(exstprods)   # confirm dataset changes
correlations12v <- cor(exstprods[,c(2,3,4,5,6,7,8,9,10,11,12)])
print(correlations12v) # evaluate correlations

# create correlation plot
corrplot(correlations12v, method = "circle")
# 80 obs of 12 vars  FourStar and PosRvw and TwoStar are the most correlated to Volume

# remove attributs from predict ds to match train ds 
newprods$FiveStar <- NULL
newprods$OneStar <- NULL
newprods$ThreeStar <- NULL
newprods$NegRvw <- NULL

exstprods12v <- exstprods
newprods12v <- newprods

str(exstprods12v) #confirm changes # 80 obs of 12 vars
str(newprods12v)  #confirm changes # 17 obs of 12 vars


##################
# Train/test sets 
##################

set.seed(123) # set random seed - random selection can be reproduced

## create and compare 16 var and 12 var train and test ds

## create the training partition that is 75% of total obs 
##  for 16 var ds
#inTraining <- createDataPartition(exstprods16v$Volume, p=0.75, list=FALSE)
#trainSet <- exstprods16v[inTraining,]   #  create training dataset for 16 vars
#testSet <- exstprods16v[-inTraining,]   #  create test/validate dataset for 16 vars

### create the training partition that is 75% of total obs 
###  for 12 var ds; comment 3 lines above and uncomment 3 lines below for the second run
inTraining <- createDataPartition(exstprods12v$Volume, p=0.75, list=FALSE)
trainSet <- exstprods12v[inTraining,]  #  create training dataset for 12 vars
testSet <- exstprods12v[-inTraining,]  #  create test/validate dataset for 12 vars

str(trainSet) # 61 obs of 16 or 12 var
str(testSet)  # 19 obs of 16 or 12 var

################
# Train control
################

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
#fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

##############
# Train and test model on train and test/validate ds
##############
GLMfit1 <- train(Volume~., data = trainSet, method = "glm", trControl = fitControl) 
## for 16v/12v trainSet received 13/14 warnings In predict.lm(object, newdata, se.fit, scale = 1, type = ifelse(type ==  ... :
## prediction from a rank-deficient fit may be misleading
## probably because 5star and Volume had correlation of 1, but 5star was removed for 12v, so what's the warning for?

LMfit1 <- train(Volume~., data = trainSet, method = "lm", trControl = fitControl) 

## for 16v/12v trainSet received 15/11 warnings In predict.lm(modelFit, newdata) :
## prediction from a rank-deficient fit may be misleading 
## probably because 5star and Volume had correlation of 1, but 5star was removed for 12v, so what's the warning for?

GLMfit1
##### this comment area is populated with the results of 16v and 12v train/test(fit) runs
## using 16 vars
## RMSE          Rsquared
## 3.351858e-13  1   
## using 12 vars
## RMSE      Rsquared 
## 371.9133  0.7703459

LMfit1
## using 16 vars
## RMSE          Rsquared
## 4.663796e-13  1     
## using 12 vars
## RMSE      Rsquared 
## 395.4184  0.7468034

GLMfit2 <- train(Volume~., data = testSet, method = "glm", trControl = fitControl) 
## for 16v and 12v ds received 50 or more warnings
## In predict.lm(object, newdata, se.fit, scale = 1, type = ifelse(type ==  ... :
## prediction from a rank-deficient fit may be misleading"  -

LMfit2 <- train(Volume~., data = testSet, method = "lm", trControl = fitControl) 
## for 16v and 12v ds received 50 or more warnings "In predict.lm(modelFit, newdata): 
## prediction from a rank-deficient fit may be misleading"  - 

GLMfit2
##### this comment area is populated with the results of 16v and 12v tests are run (after the run)
## using 16 vars
## RMSE          Rsquared
## 1.905865e-09  1       

## using 12 vars
## RMSE      Rsquared
## 2079.636  1    

LMfit2
### using 16 vars
### RMSE          Rsquared
### 1.322953e-09  1

### using 12 vars
### RMSE      Rsquared
### 1960.053  1

summary(GLMfit1)
## for 16v
## Residual standard error: 1.549e-13 on 35 degrees of freedom
## Multiple R-squared:      1,	Adjusted R-squared:      1 
## F-statistic: 3.858e+31 on 25 and 35 DF,  p-value: < 2.2e-16

## for 12V
## Null deviance: 23137303  on 60  degrees of freedom
## Residual deviance:  2208165  on 39  degrees of freedom
## AIC: 859.42

summary(LMfit1) ## for 16v warning: "essentially perfect fit: summary may be unreliable"
                ## Residual standard error: 1.549e-13 on 35 degrees of freedom
                ## Multiple R-squared:      1,	Adjusted R-squared:      1 
                ## F-statistic: 3.858e+31 on 25 and 35 DF,  p-value: < 2.2e-16

                ## for 12v Residual standard error: 237.9 on 39 degrees of freedom
                ## Multiple R-squared:  0.9046,	Adjusted R-squared:  0.8532
                ## F-statistic: 17.6 on 21 and 39 DF,  p-value: 7.44e-14

summary(GLMfit2)
                ## for 16v
                ## Deviance Residuals: 
                ## [1]  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
                ## Coefficients: (7 not defined because of singularities)
                ## Null deviance: 1.4837e+08  on 18  degrees of freedom
                ## Residual deviance: 2.2484e-23  on  0  degrees of freedom
                ## AIC: -952.86
                ## Number of Fisher Scoring iterations: 1

                ## for 12v 
                ## Null deviance: 1.4837e+08  on 18  degrees of freedom
                ## Residual deviance: 4.0352e+02  on  1  degrees of freedom
                ## AIC: 149.98

summary(LMfit2)
                ## for 16v, ALL 19 residuals are 0: no residual degrees of freedom!
                ## probably not enough sample, 7 coefficients not defined becuase of singularities
                ## more variables than samples, I think ???
                ## Multiple R-squared:      1,	Adjusted R-squared:    NaN 
                ## F-statistic:   NaN on 18 and 0 DF,  p-value: NA  

                ## for 12v ALL 19 residuals are 0: no residual degrees of freedom!
                ## Multiple R-squared:      1,	Adjusted R-squared:    1
                ## F-statistic:   2.163e+04 on 17 and 1 DF,  p-value: 0.005346
# estimate variable importance

GLMvarimp1 <- varImp(GLMfit1, scale = FALSE)
GLMvarimp2 <- varImp(GLMfit2, scale = FALSE)

LMvarimp1 <- varImp(LMfit1, scale = FALSE)  # for 16v perfect fit: summary may be unreliable
LMvarimp2 <- varImp(LMfit2, scale = FALSE)  

# summarize importance
print(GLMvarimp1)
           ## for 16v, 5star is much higher than others
           ## for 12v, 4star, ProdDepth higher than others
print(GLMvarimp2) 
           ## 18 warnings for 16v 
           ## In FUN(newX[, i], ...) : no non-missing arguments to max; returning -Inf
           ## for 12v PosRvw and ProdTypeWarranty are much higher than others

print(LMvarimp1)  ## for 16v, 5star is much higher than others
                  ## for 12v, 4 star and prodDepth much higher than others 
print(LMvarimp2)  ## 18 warnings for 16v
                  ## "In FUN(newX[, i], ...) : no non-missing arguments to max; returning -Inf"
                  ## for 12v PosRvw and ProdTypeExtendedWarranty are much higher
RFfit1 <- train(Volume~., data = trainSet, method = "rf", trControl = fitControl) 
RFfit2 <- train(Volume~., data = testSet, method = "rf", trControl = fitControl) 
    ## warning msg for 16v and 12v
    ## "Warning message: In nominalTrainWorkflow(x = x, y = y, wts = weights, info = trainInfo,  :
    ##                  There were missing values in resampled performance measures."
RFfit1
####using 16 vars
### mtry  RMSE       Rsquared 
### 2    248.20309  0.9213909
### 13    115.25675  0.9788870
### 25     86.95611  0.9869556  <-  this one is the best of the three

####using 12 vars 
### mtry  RMSE      Rsquared 
###   2    314.7895  0.8601197
###  11    209.8869  0.9122668
###  21    207.0788  0.9101674  <- this one was used 

RFfit2
### using 16 vars
### mtry  RMSE       Rsquared
###  2    1372.584   1       
### 13    1009.660   1  <-  this one the final      
### 25    1042.850   1 
   
### using 12 vars
### mtry  RMSE      Rsquared
###  2    1711.243  1       
### 11    1144.776  1       
### 21    1067.244  1   <-  this one was used         

summary(RFfit1)
summary(RFfit2)

# estimate variable importance
################this section doesn't work for rf ##################
#RFvarimp1 <- varImp(RFfit1, scale = FALSE)
#RFvarimp2 <- varImp(RFfit2, scale = FALSE)

## summarize importance
#print(RFvarimp1)
#print(RFvarimp2)
##################################
SVMfit1 <- train(Volume~., data = trainSet, method = "svmLinear2", trControl = fitControl) 
   ### WARNING: reaching max number of iterations   
   ### for 16v/12v - 36/33 warnings:
   ### In svm.default(x = as.matrix(x), y = y, kernel = "linear",  ... :
   ### Variable(s) 'ProductTypeGame.Console' constant. Cannot scale data.
   ###             'ProductTypePrinter.Supplies'
  ###              'ProductTypePC'
   ### for 12v - 13 WARNINGs: reaching max number of iterations
   ### 33 warnings "In svm.default(x = as.matrix(x), y = y, kernel = "linear",  ... :
   ### Variable(s) 'ProductTypeGame.Console' constant. Cannot scale data."

SVMfit2 <- train(Volume~., data = testSet, method = "svmLinear2", trControl = fitControl)
   ### for 16v warning:reaching max number of iterations
   ### for 12v 50+ warnings
   ### "In svm.default(x = as.matrix(x), y = y, kernel = "linear",  ... :
   ### Variable(s) 'ProductTypeLaptop' and 'ProductTypeNetbook' and 
   ### 'ProductTypeSoftware' and 'ProductTypeTablet' constant. Cannot scale data."
SVMfit1
### using 16 vars
### cost  RMSE      Rsquared 
### 0.25  95.41911  0.9770900
### 0.50  90.89453  0.9786056   <- this one was used
### 1.00  90.89453  0.9786056

### using 12 vars
### cost  RMSE      Rsquared 
### 0.25  339.1376  0.8514334 <-  this one was used
### 0.50  376.8540  0.8052877
### 1.00  383.3874  0.7875312

SVMfit2
### using 16 vars
### cost  RMSE       Rsquared
### 0.25  0.5803712  1     <-  this one was used   
### 0.50  0.5803712  1       
### 1.00  0.5803712  1  

### using 12 vars
### cost  RMSE      Rsquared
### 0.25  759.3397  1 <-  this one was used       
### 0.50  765.8649  1      
### 1.00  860.8087  1           

summary(SVMfit1)
summary(SVMfit2)

# estimate variable importance
SVMvarimp1 <- varImp(SVMfit1, scale = FALSE)
SVMvarimp2 <- varImp(SVMfit2, scale = FALSE)  ## not enough sampling obs.?
### for 16v/12v Error: $ operator is invalid for atomic vectors
### In addition: Warning message:
###  In storage.mode(x) <- "double" : NAs introduced by coercion

# summarize importance
print(SVMvarimp1)  # for 16v ds --5star, PosRvw, 4star, 
                   # for 12v ds -- PosRvw and 4Star were the highest
print(SVMvarimp2)  # no output for 16v/12v, object not found
# plot importance

#predictor variables
predictors(LMfit1)  
predictors(LMfit2) 
predictors(RFfit1)  
predictors(RFfit2)  
predictors(SVMfit1)  
predictors(SVMfit2)  

######## run this part after deciding which model is the bests (and running it), in this case 12 var RF
# save model object -- choose the best model
#saveRDS(RFfit1, "salesvolumes_RF16.rds")   #saved in "C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/R/Sales"
#saveRDS(SVMfit1, "salesvolumes_SVM16.rds")
#saveRDS(LMfit1, "salesvolumes_LM16.rds")
#saveRDS(GLMfit1, "salesvolumes_GLM16.rds")

# 12v models
saveRDS(RFfit1, "salesvolumes_RF12.rds") #saved in "C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/R/Sales"
saveRDS(SVMfit1, "salesvolumes_SVM12.rds")
saveRDS(LMfit1, "salesvolumes_LM12.rds")
saveRDS(GLMfit1, "salesvolumes_GLM12.rds")
########################
## Predict using test ds  
########################
# load and name model
#modeltest <- readRDS("salesvolumes_RF16.rds")
#modeltest <- readRDS("salesvolumes_SVM16.rds")
#modeltest <- readRDS("salesvolumes_LM16.rds")
#modeltest <- readRDS("salesvolumes_GLM16.rds")

modeltest <- readRDS("salesvolumes_RF12.rds")
#modeltest <- readRDS("salesvolumes_SVM12.rds")
#modeltest <- readRDS("salesvolumes_LM12.rds")
#modeltest <- readRDS("salesvolumes_GLM12.rds")

modelpredtest <- predict(modeltest, testSet)  # predict on valide ds with trained model 
modelpredtest # output predicted values for each obs 
#plot(modelpredtest)

#head(modelpredtest)
#plot predicted verses actual
plot(modelpredtest)

comparison <- cbind(testSet$Volume, modelpredtest)
colnames(comparison) <- c("actual","predicted") 
print(comparison)
#head(comparison)
summary(comparison)
mape <- (sum(abs(comparison[,1] - comparison[,2])/abs(comparison[,1]))/nrow(comparison))*100
mape
  ### RF 16v - 16.33; 12v - 35.42
  ### SVM 16v -      ; 12v - 126.22
  ### GLM 16v -      ; 12v - 336.80
  ### LM  16v -      ; 12v - 336.80  
mapeTable <- cbind(comparison,abs(comparison[,1]-comparison[,2])/comparison[,1]*100)
colnames(mapeTable)[3] <- "absolute percent error"
#head(mapeTable)
mapeTable

#######################
# Predict with predict set
#######################
# load and name model
#model <- readRDS("salesvolumes_RF16.rds")
#model <- readRDS("salesvolumes_SVM16.rds")
#model <- readRDS("salesvolumes_LM16.rds")
#model <- readRDS("salesvolumes_GLM16.rds")

model <- readRDS("salesvolumes_RF12.rds")
#model <- readRDS("salesvolumes_SVM12.rds")
#model <- readRDS("salesvolumes_LM12.rds")
#model <- readRDS("salesvolumes_GLM12.rds")
# use one(appropriate) of the ds below
#predictSet16v <- newprods16v
predictSet12v <- newprods12v

#str(predictSet16v) #17 obs of 16 var
str(predictSet12v) #17 obs of 12 var

#modelpred <- predict(model, predictSet16v)  # predict on predictSet 16v with trained model 
modelpred <- predict(model, predictSet12v)  # predict on predictSet 12v with trained model 
modelpred # output predicted values for each obs 
plot(modelpred)

