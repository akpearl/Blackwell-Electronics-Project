# Project name: Computer Brand Preference
# File name:    Survey Complete Responses (75% - train and 15% -test ds) 
#               Survey Incomplete (predict ds)      
###############
# Housekeeping
###############

rm(list = ls()) # Clear objects if necessary
getwd()         # find out where the working directory is currently set
                # set working directory 
setwd("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/R/Brand Preference Prediction")

################
# Load packages
################
install.packages("party") 
install.packages("caret")
install.packages("corrplot")
install.packages("randomForest") 

require(party)
require(caret)  
require(corrplot)
require(randomForest)

#######----------------------------------------#######
# Import data
#######----------------------------------------#######

#read train file
complete <- read.csv("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/R/Brand Preference Prediction/Survey_Key_and_Complete_Responses.csv", 
            header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)
#read predict file
incomplete <- read.csv("C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/R/Brand Preference Prediction/SurveyIncomplete.csv", 
            header = TRUE, check.names=FALSE,sep = ",", quote = "\'", as.is = TRUE)
# view data structure                
str(complete)  # 10,000 obs of 7 vars 
str(incomplete)   # 5,000 obs of 7 vars 

write.csv(complete, "complete_survey.csv") # save copy of original ds in working directory
write.csv(incomplete, "incomplete_survey.csv") # save copy of original ds in working directory

######-----------------------------------------#######
# Evaluate data
######-----------------------------------------####### 

head(complete,5)
head(incomplete,5)

tail(complete,5)
tail(incomplete,5)

summary(complete)
summary(incomplete)

######------------------------------------------#######
# Pre-process
######------------------------------------------#######

any(is.na(complete))  # confirm if any "NA" values in ds
any(is.na(incomplete))   # confirm if any "NA" values in ds

## use this code to remove obvious attributes
## complete$??? <- NULL
#########  -- all variables are included in this analysis

## use this code to change data types to ?? (integer/factor)
## complete$age <- as.integer(complete$age)

# change data types to factor
complete$elevel <- factor(complete$elevel)
complete$car <- factor(complete$car)
complete$zipcode <- factor(complete$zipcode)
complete$brand <- factor(complete$brand)

# change data types so train and predict data sets match 
incomplete$elevel <- factor(incomplete$elevel)   ###  education level code doesn't have numeric meaning, is not intended for calculations
incomplete$car <- factor(incomplete$car)         ###  car brand code doesn't have numeric meaning, is not intended for calculations
incomplete$zipcode <- factor(incomplete$zipcode) ###  zipcode doesn't have numeric meaning, is not intended for calculations
incomplete$brand <- factor(incomplete$brand)     ###  brand code doesn't have numeric meaning, is not intended for calculations

correlations <- cor(incomplete[,c(1,6)])  ### see correlations between 6 variables, #7 variable is to be predicted
print(correlations)   

complete7v <- complete
incomplete7v <- incomplete

str(complete7v) #confirm changes # 10,000 obs of 7 vars
str(incomplete7v)  #confirm changes # 5,000 obs of 7 vars

##########--------------------------------------------########
# Create Train/Test sets 
##########--------------------------------------------########

set.seed(123) # set random seed (random selection can be reproduced)

## create the training partition that is 75% of total obs 
inTraining <- createDataPartition(complete7v$brand, p=0.75, list=FALSE)
trainSet <- complete7v[inTraining,]   #  create training dataset 
testSet <- complete7v[-inTraining,]   #  create test/validate dataset

str(trainSet) # 7501 obs of 7 var
str(testSet) # 2499 obs of 7 var

#########--------------------------------------------#######
# Train control
#########--------------------------------------------#######

fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 10)
####fitControl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

#########--------------------------------------------########
# Train and test model on train ds
########---------------------------------------------########
ctreefit1 <- train(brand~., data = trainSet, method = "ctree", trControl = fitControl) 
ctreefit1
   ## mincriterion  Accuracy   Kappa    
   ## 0.01          0.9128516  0.8149954 <- this one was used
   ## 0.50          0.9004560  0.7921057
   ## 0.99          0.8404618  0.6800241
plot(ctreefit1)

RFfit1 <- train(brand~., data = trainSet, method = "rf", trControl = fitControl)
RFfit1
     ## mtry  Accuracy   Kappa       
   ##  2    0.6216638  4.393118e-05
   ## 18    0.9194511  8.288257e-01 <- this one was used
   ## 34    0.9164118  8.224934e-01  
plot(RFfit1)

KNNfit1 <- train(brand~., data = trainSet, method = "knn", trControl = fitControl) 
KNNfit1
   ## k  Accuracy   Kappa    
   ## 5  0.6782297  0.3122591
   ## 7  0.6870822  0.3306579
   ## 9  0.6952685  0.3487749  <- this one was used
plot(KNNfit1)

summary(ctreefit1)
summary(RFfit1)
summary(KNNfit1)

#### estimate variable importance -- note how results differ between ctree, RF, and KNN

ctreevarimp1 <- varImp(ctreefit1, scale = FALSE)
RFvarimp1 <- varImp(RFfit1, scale = FALSE)  
KNNvarimp1 <- varImp(KNNfit1, scale = FALSE)  
 
#### summarize importance
print(ctreevarimp1)
  ## Importance
  ## salary      0.6211
  ## credit      0.5111
  ## age         0.5110
  ## elevel      0.5050
  ## zipcode     0.5046
  ## car         0.5021       

print(RFvarimp1)   
# only 20 most important variables shown (out of 34)
# salary   1744.80
# age      1224.93
# credit    179.30
# elevel4    19.26
# elevel1    18.61
# elevel3    18.12
# elevel2    16.89
# zipcode4   15.94
# zipcode7   15.01
# zipcode6   14.67
# zipcode3   14.38
# zipcode1   13.84
# zipcode5   13.76
# zipcode2   13.57
# zipcode8   13.38
# car7       13.29
# car12      12.39
# car5       11.20
# car14      10.75
# car10      10.68 

print(KNNvarimp1)  
# Importance
# salary      0.6211
# credit      0.5111
# age         0.5110
# elevel      0.5050
# zipcode     0.5046
# car         0.5021  


#####  predictor variables
predictors(ctreefit1)  
predictors(RFfit1)  
predictors(KNNfit1)  

## save model object -- 
## saved in "C:/Users/AntoninaPearl/Downloads/Rutgers Data Analytics/R/Brand Preference Prediction"
saveRDS(ctreefit1, "BrandPreference_ctree.rds")   
saveRDS(RFfit1, "BrandPreference_RF.rds")
saveRDS(KNNfit1, "BrandPreference_KNN.rds")

##########--------------------------------------------#########
## Predict using TRAIN ds  
##########--------------------------------------------#########
### load and name model --  re-run this section comment/uncomment models
### modeltest <- readRDS("BrandPreference_ctree.rds")
modeltest <- readRDS("BrandPreference_RF.rds")
#### modeltest <- readRDS("BrandPreference_KNN.rds")

modelpredtest <- predict(modeltest, trainSet)  # predict on valid ds with trained model 
head(modelpredtest,50) # output predicted values for each obs
tail(modelpredtest,50) #
plot(modelpredtest)

### plot predicted verses actual

comparison <- cbind(trainSet$brand, modelpredtest)
colnames(comparison) <- c("actual","predicted") 
#print(comparison)
head(comparison)
confusionMatrix(data = modelpredtest, trainSet$brand)
###########--------------------------------------------###########
#  model: ctree stats (train ds)
###########--------------------------------------------###########
# Prediction    0    1
#            0 1512 1416
#            1 1326 3247
# 
# Accuracy : 0.6344          
# 95% CI : (0.6234, 0.6454)
# No Information Rate : 0.6217          
# P-Value [Acc > NIR] : 0.01137         
# 
# Kappa : 0.2277          
# Mcnemar's Test P-Value : 0.08920         
# 
# Sensitivity : 0.5328          
# Specificity : 0.6963          
# Pos Pred Value : 0.5164          
# Neg Pred Value : 0.7100          
# Prevalence : 0.3783          
# Detection Rate : 0.2016          
# Detection Prevalence : 0.3903          
# Balanced Accuracy : 0.6146          
# 
# 'Positive' Class : 0      
##############--------------------------------------------###########
#  model: rf stats (train ds)
##############--------------------------------------------###########
# Prediction    0    1
#            0 2838  0
#            1  0    4663
# 
# Accuracy : 1          
# 95% CI : (0.9995, 1)
# No Information Rate : 0.6217     
# P-Value [Acc > NIR] : < 2.2e-16  
# 
# Kappa : 1          
# Mcnemar's Test P-Value : NA         
#                                      
#             Sensitivity : 1.0000     
#             Specificity : 1.0000     
#          Pos Pred Value : 1.0000     
#          Neg Pred Value : 1.0000     
#              Prevalence : 0.3783     
#          Detection Rate : 0.3783     
#    Detection Prevalence : 0.3783     
#       Balanced Accuracy : 1.0000     
#                                      
#        'Positive' Class : 0      
##############--------------------------------------------###########
#  model knn stats (train ds)
##############--------------------------------------------###########
# Prediction    0    1
#            0 1876  858
#            1  962 3805
# 
# Accuracy : 0.7574         
# 95% CI : (0.7475, 0.767)
# No Information Rate : 0.6217         
# P-Value [Acc > NIR] : < 2e-16        
# 
# Kappa : 0.4805         
# Mcnemar's Test P-Value : 0.01576        
# 
# Sensitivity : 0.6610         
# Specificity : 0.8160         
# Pos Pred Value : 0.6862         
# Neg Pred Value : 0.7982         
# Prevalence : 0.3783         
# Detection Rate : 0.2501         
# Detection Prevalence : 0.3645         
# Balanced Accuracy : 0.7385         
# 
# 'Positive' Class : 0   
###########--------------------------------------------###########
## Predict using TEST ds  
###########--------------------------------------------###########
# load and name  -- re-run this section comment/uncomment models
#modeltest <- readRDS("BrandPreference_ctree.rds")
modeltest <- readRDS("BrandPreference_RF.rds")
#modeltest <- readRDS("BrandPreference_KNN.rds")

modelpredtest <- predict(modeltest, testSet)  # predict on valide ds with trained model 
head(modelpredtest,50) # output predicted values 
tail(modelpredtest,50)
plot(modelpredtest)

#head(modelpredtest)
#plot predicted verses actual

comparison <- cbind(testSet$brand, modelpredtest)
colnames(comparison) <- c("actual","predicted") 
#print(comparison)
head(comparison)
confusionMatrix(data = modelpredtest, testSet$brand)
#################################
#  model rf (test ds)
#################################
# Prediction    0    1
#          0  824   80
#          1  121 1474
# 
# Accuracy : 0.9196          
# 95% CI : (0.9082, 0.9299)
# No Information Rate : 0.6218          
# P-Value [Acc > NIR] : < 2.2e-16       
# 
# Kappa : 0.8275          
# Mcnemar's Test P-Value : 0.004782        
# 
# Sensitivity : 0.8720          
# Specificity : 0.9485          
# Pos Pred Value : 0.9115          
# Neg Pred Value : 0.9241          
# Prevalence : 0.3782          
# Detection Rate : 0.3297          
# Detection Prevalence : 0.3617          
# Balanced Accuracy : 0.9102          
# 
# 'Positive' Class : 0   
############################
#  model ctree (test ds)
############################
# Prediction    0    1
#          0  493  474
#          1  452 1080
# Accuracy : 0.6295          
# 95% CI : (0.6102, 0.6484)
# No Information Rate : 0.6218          
# P-Value [Acc > NIR] : 0.2229          
# 
# Kappa : 0.2157          
# Mcnemar's Test P-Value : 0.4901          
#                                           
#             Sensitivity : 0.5217          
#             Specificity : 0.6950          
#          Pos Pred Value : 0.5098          
#          Neg Pred Value : 0.7050          
#              Prevalence : 0.3782          
#          Detection Rate : 0.1973          
#    Detection Prevalence : 0.3870          
#       Balanced Accuracy : 0.6083          
#                                           
#        'Positive' Class : 0  
############################
#  model knn (test ds)
############################
# Prediction    0    1
#           0  548  370
#           1  397 1184
# 
# Accuracy : 0.6931          
# 95% CI : (0.6746, 0.7111)
# No Information Rate : 0.6218          
# P-Value [Acc > NIR] : 5.546e-14       
# 
# Kappa : 0.3437          
# Mcnemar's Test P-Value : 0.3478          
# 
# Sensitivity : 0.5799          
# Specificity : 0.7619          
# Pos Pred Value : 0.5969          
# Neg Pred Value : 0.7489          
# Prevalence : 0.3782          
# Detection Rate : 0.2193          
# Detection Prevalence : 0.3673          
# Balanced Accuracy : 0.6709          
# 
# 'Positive' Class : 0 
#########---------------------------------------------##########
# resample 
#########---------------------------------------------##########
resamps <- resamples(list(ctree = ctreefit1,rf = RFfit1, knn = KNNfit1))
summary(resamps)
# Models: ctree, rf, knn 
# Number of resamples: 100 
# 
# Accuracy 
#          Min.   1st Qu.  Median      Mean   3rd Qu.      Max.    NA's
# ctree 0.8733333 0.9051265 0.91600 0.9128516 0.9213595 0.9345794    0
# rf    0.8973333 0.9120879 0.92000 0.9194511 0.9253333 0.9519359    0
# knn   0.6537949 0.6840000 0.69487 0.6952685 0.7060000 0.7356475    0
# 
# Kappa 
#            Min.   1st Qu.    Median      Mean   3rd Qu.      Max. NA's
# ctree 0.7342865 0.7988214 0.8206202 0.8149954 0.8341410 0.8609489    0
# rf    0.7804600 0.8139757 0.8305243 0.8288257 0.8410354 0.8970549    0
# knn   0.2638809 0.3262514 0.3459807 0.3487749 0.3708430 0.4290147    0
diffs <- diff(resamps)
summary(diffs)
# Accuracy 
# ctree     rf        knn    
# ctree           -0.0066    0.2176
# rf    0.0007823            0.2242
# knn   < 2.2e-16 < 2.2e-16        
# 
# Kappa 
# ctree     rf        knn     
# ctree           -0.01383   0.46622
# rf    0.0008215            0.48005
# knn   < 2.2e-16 < 2.2e-16  
#############-----------------------------------------##########
# Predict with predict set
#############-----------------------------------------##########
######## run this part after deciding which model is the bests 
# load and name model
#model <- readRDS("BrandPreference_ctree.rds")
model <- readRDS("BrandPreference_RF.rds")
#model <- readRDS("BrandPreference_KNN.rds")

predictSet <- incomplete7v

str(predictSet) #5000 obs of 7 var

modelpred <- predict(model, predictSet) 
head(modelpred,50) # output predicted values for each obs 
tail(modelpred,50)
plot(modelpred)

output <-  cbind(modelpred, incomplete7v)
#write.csv(output, file = "incompletectree.csv", row.names = FALSE)
write.csv(output, file = "incompleteRF.csv", row.names = FALSE)
#write.csv(output, file = "incompleteKNN.csv", row.names = FALSE)


