# Coursera – Practical Machine Learning assignment
vleonov  
21 Mar 2015  

> ### Background
>
>Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 
>
>
> ###Data 
>
>The training data for this project are available here: 
>
>https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
>
>The test data are available here: 
>
>https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
>
>The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment. 
>
> ###What you should submit
>
>The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 
>
>1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
>2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 
>
> ###Reproducibility 
>
>Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis. 

#Intro

We will try to predict exercide class, using data recorded from several people using several sensors. We will user `Random Forest Modeling`, which gives us over 99% accuracy on the training set of 75% of the total data. Moreover, in `Random Forest` there is no need for cross-validation, because it estimated internally, see http://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#ooberr

#Prepare the enviroment

Load required libraries and set seed for reproducibility.


```r
library(data.table)
library(caret)
library(parallel)
library(doParallel)

set.seed(2103)
```

#Prepare the dataset

Read the training and testing (actually validating) data.


```r
data <- read.csv(file="data/pml-training.csv", head=TRUE, sep=",", na.strings = c("", "NA", "#DIV/0!"))
validation <- read.csv(file="data/pml-testing.csv", head=TRUE, sep=",", na.strings = c("", "NA", "#DIV/0!"))
```

We need only sensors' data for prediction. Moreover we exclude data, which contains NA values.


```r
isNA <- function(x) { any(is.na(x))}
hasNA <- sapply(data, excludeNA)

isPredictor <- !hasNA & grepl("belt|forearm|arm|dumbbell", names(hasNA))
predictors <- names(data)[isPredictor]
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"
```

Subset the dataset to include only predictors and outcome variable `classe`.

```r
data <- data[,c('classe', predictors)]
```

Convert `classe` to factor.

```r
data$classe <- factor(data$classe)
```

It's time to divide dataset to training and testing partitions.

```r
inTrain <- createDataPartition(data$classe, p=0.75, list=FALSE)
training <- data[inTrain,]
testing <- data[-inTrain,]
```

Let's see what we have.

```r
summary(training)
```

```
##  classe     roll_belt        pitch_belt          yaw_belt      
##  A:4185   Min.   :-28.60   Min.   :-55.8000   Min.   :-180.00  
##  B:2848   1st Qu.:  1.10   1st Qu.:  1.7500   1st Qu.: -88.30  
##  C:2567   Median :113.00   Median :  5.2600   Median : -13.10  
##  D:2412   Mean   : 64.33   Mean   :  0.2962   Mean   : -11.31  
##  E:2706   3rd Qu.:123.00   3rd Qu.: 15.2000   3rd Qu.:  12.50  
##           Max.   :162.00   Max.   : 60.3000   Max.   : 179.00  
##  total_accel_belt  gyros_belt_x        gyros_belt_y       gyros_belt_z  
##  Min.   : 1.0     Min.   :-1.040000   Min.   :-0.51000   Min.   :-1.46  
##  1st Qu.: 3.0     1st Qu.:-0.030000   1st Qu.: 0.00000   1st Qu.:-0.20  
##  Median :17.0     Median : 0.030000   Median : 0.02000   Median :-0.10  
##  Mean   :11.3     Mean   :-0.005197   Mean   : 0.03939   Mean   :-0.13  
##  3rd Qu.:18.0     3rd Qu.: 0.110000   3rd Qu.: 0.11000   3rd Qu.:-0.02  
##  Max.   :29.0     Max.   : 2.220000   Max.   : 0.63000   Max.   : 1.62  
##   accel_belt_x       accel_belt_y     accel_belt_z    magnet_belt_x   
##  Min.   :-120.000   Min.   :-54.00   Min.   :-275.0   Min.   :-52.00  
##  1st Qu.: -21.000   1st Qu.:  3.00   1st Qu.:-162.0   1st Qu.:  9.00  
##  Median : -14.000   Median : 33.50   Median :-152.0   Median : 35.00  
##  Mean   :  -5.545   Mean   : 30.13   Mean   : -72.5   Mean   : 55.76  
##  3rd Qu.:  -5.000   3rd Qu.: 61.00   3rd Qu.:  27.0   3rd Qu.: 60.00  
##  Max.   :  85.000   Max.   :164.00   Max.   : 103.0   Max.   :485.00  
##  magnet_belt_y   magnet_belt_z       roll_arm         pitch_arm      
##  Min.   :354.0   Min.   :-623.0   Min.   :-180.00   Min.   :-88.800  
##  1st Qu.:581.0   1st Qu.:-375.0   1st Qu.: -31.60   1st Qu.:-26.200  
##  Median :601.0   Median :-320.0   Median :   0.00   Median :  0.000  
##  Mean   :593.5   Mean   :-345.8   Mean   :  18.04   Mean   : -4.654  
##  3rd Qu.:610.0   3rd Qu.:-306.0   3rd Qu.:  77.60   3rd Qu.: 11.500  
##  Max.   :673.0   Max.   : 287.0   Max.   : 180.00   Max.   : 88.500  
##     yaw_arm         total_accel_arm  gyros_arm_x        gyros_arm_y     
##  Min.   :-180.000   Min.   : 1.00   Min.   :-6.37000   Min.   :-3.4400  
##  1st Qu.: -43.100   1st Qu.:17.00   1st Qu.:-1.35000   1st Qu.:-0.8000  
##  Median :   0.000   Median :27.00   Median : 0.08000   Median :-0.2400  
##  Mean   :  -0.435   Mean   :25.51   Mean   : 0.03632   Mean   :-0.2559  
##  3rd Qu.:  46.200   3rd Qu.:33.00   3rd Qu.: 1.57000   3rd Qu.: 0.1600  
##  Max.   : 180.000   Max.   :66.00   Max.   : 4.87000   Max.   : 2.8100  
##   gyros_arm_z       accel_arm_x       accel_arm_y       accel_arm_z     
##  Min.   :-2.3300   Min.   :-404.00   Min.   :-315.00   Min.   :-636.00  
##  1st Qu.:-0.0700   1st Qu.:-241.00   1st Qu.: -54.00   1st Qu.:-144.00  
##  Median : 0.2300   Median : -44.00   Median :  15.00   Median : -49.00  
##  Mean   : 0.2699   Mean   : -59.89   Mean   :  32.64   Mean   : -71.64  
##  3rd Qu.: 0.7200   3rd Qu.:  84.00   3rd Qu.: 139.00   3rd Qu.:  23.00  
##  Max.   : 3.0200   Max.   : 431.00   Max.   : 308.00   Max.   : 292.00  
##   magnet_arm_x     magnet_arm_y     magnet_arm_z    roll_dumbbell    
##  Min.   :-580.0   Min.   :-392.0   Min.   :-597.0   Min.   :-153.71  
##  1st Qu.:-299.0   1st Qu.: -13.0   1st Qu.: 121.2   1st Qu.: -18.46  
##  Median : 290.0   Median : 201.0   Median : 443.5   Median :  48.02  
##  Mean   : 192.5   Mean   : 155.7   Mean   : 305.2   Mean   :  23.61  
##  3rd Qu.: 638.0   3rd Qu.: 322.0   3rd Qu.: 546.0   3rd Qu.:  67.40  
##  Max.   : 780.0   Max.   : 582.0   Max.   : 694.0   Max.   : 153.55  
##  pitch_dumbbell     yaw_dumbbell      total_accel_dumbbell
##  Min.   :-149.59   Min.   :-148.766   Min.   : 0.00       
##  1st Qu.: -40.14   1st Qu.: -77.613   1st Qu.: 4.00       
##  Median : -20.90   Median :  -3.207   Median :10.00       
##  Mean   : -10.69   Mean   :   1.953   Mean   :13.68       
##  3rd Qu.:  17.30   3rd Qu.:  80.643   3rd Qu.:19.00       
##  Max.   : 149.40   Max.   : 154.952   Max.   :58.00       
##  gyros_dumbbell_x    gyros_dumbbell_y   gyros_dumbbell_z  
##  Min.   :-204.0000   Min.   :-2.10000   Min.   : -2.3800  
##  1st Qu.:  -0.0300   1st Qu.:-0.14000   1st Qu.: -0.3100  
##  Median :   0.1300   Median : 0.05000   Median : -0.1300  
##  Mean   :   0.1561   Mean   : 0.04866   Mean   : -0.1229  
##  3rd Qu.:   0.3500   3rd Qu.: 0.21000   3rd Qu.:  0.0300  
##  Max.   :   2.2200   Max.   :52.00000   Max.   :317.0000  
##  accel_dumbbell_x  accel_dumbbell_y  accel_dumbbell_z  magnet_dumbbell_x
##  Min.   :-419.00   Min.   :-189.00   Min.   :-334.00   Min.   :-643.0   
##  1st Qu.: -50.00   1st Qu.:  -8.00   1st Qu.:-141.00   1st Qu.:-536.0   
##  Median :  -8.00   Median :  42.00   Median :  -1.00   Median :-480.0   
##  Mean   : -28.41   Mean   :  52.37   Mean   : -37.69   Mean   :-328.4   
##  3rd Qu.:  11.00   3rd Qu.: 109.00   3rd Qu.:  38.00   3rd Qu.:-304.0   
##  Max.   : 235.00   Max.   : 310.00   Max.   : 318.00   Max.   : 584.0   
##  magnet_dumbbell_y magnet_dumbbell_z  roll_forearm     pitch_forearm   
##  Min.   :-744.0    Min.   :-262.00   Min.   :-180.00   Min.   :-72.50  
##  1st Qu.: 231.0    1st Qu.: -46.00   1st Qu.:  -1.27   1st Qu.:  0.00  
##  Median : 310.0    Median :  14.00   Median :  20.35   Median :  9.35  
##  Mean   : 220.5    Mean   :  46.85   Mean   :  33.46   Mean   : 10.69  
##  3rd Qu.: 389.0    3rd Qu.:  96.00   3rd Qu.: 140.00   3rd Qu.: 28.30  
##  Max.   : 633.0    Max.   : 452.00   Max.   : 180.00   Max.   : 89.80  
##   yaw_forearm      total_accel_forearm gyros_forearm_x  
##  Min.   :-180.00   Min.   :  0.00      Min.   :-22.000  
##  1st Qu.: -68.00   1st Qu.: 29.00      1st Qu.: -0.220  
##  Median :   0.00   Median : 36.00      Median :  0.050  
##  Mean   :  19.62   Mean   : 34.75      Mean   :  0.157  
##  3rd Qu.: 110.00   3rd Qu.: 41.00      3rd Qu.:  0.560  
##  Max.   : 180.00   Max.   :108.00      Max.   :  3.480  
##  gyros_forearm_y     gyros_forearm_z    accel_forearm_x  
##  Min.   : -6.62000   Min.   : -8.0900   Min.   :-498.00  
##  1st Qu.: -1.45000   1st Qu.: -0.1800   1st Qu.:-179.00  
##  Median :  0.03000   Median :  0.0800   Median : -57.00  
##  Mean   :  0.08231   Mean   :  0.1547   Mean   : -62.67  
##  3rd Qu.:  1.64000   3rd Qu.:  0.4900   3rd Qu.:  75.00  
##  Max.   :311.00000   Max.   :231.0000   Max.   : 477.00  
##  accel_forearm_y   accel_forearm_z   magnet_forearm_x   magnet_forearm_y
##  Min.   :-632.00   Min.   :-446.00   Min.   :-1280.00   Min.   :-892.0  
##  1st Qu.:  55.25   1st Qu.:-181.00   1st Qu.: -617.00   1st Qu.:   2.0  
##  Median : 201.00   Median : -38.00   Median : -382.00   Median : 590.0  
##  Mean   : 163.27   Mean   : -54.12   Mean   : -314.18   Mean   : 379.6  
##  3rd Qu.: 312.00   3rd Qu.:  27.00   3rd Qu.:  -75.25   3rd Qu.: 737.0  
##  Max.   : 923.00   Max.   : 291.00   Max.   :  672.00   Max.   :1460.0  
##  magnet_forearm_z
##  Min.   :-973.0  
##  1st Qu.: 180.2  
##  Median : 509.0  
##  Mean   : 390.0  
##  3rd Qu.: 653.0  
##  Max.   :1090.0
```

Looks like we need to preprocess data. Do it by centering and scaling.

```r
preProc <- preProcess(training[,c(predictors)])
```

Apply this preprocessing to all datasets.

```r
training <- data.table(data.frame(classe = training$classe, predict(preProc, training[,predictors])))
testing <- data.table(data.frame(classe = testing$classe, predict(preProc, testing[,predictors])))
validation <- data.table(data.frame(problem_id = validation$problem_id, predict(preProc, validation[,predictors])))
```

#Training

It's a good idea to use all available cores and process random forests in parallel.

```r
cl <- makeCluster(detectCores() )
registerDoParallel(cl)
ctrl <- trainControl(classProbs=TRUE, savePredictions=TRUE, allowParallel=T)
```

Let's start computation. It will take several minutes, good time to drink a cup of cofee.

```r
modFit <- train(classe ~ ., method="rf", data=training, trControl=ctrl)
save(modFit, file="modFit.RData")
```


Stop the clusters for parallel.

```r
stopCluster(cl)
```

#Testing


```r
confusionMatrix(predict(modFit, testing), testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    0    0    0    0
##          B    1  948    2    0    0
##          C    0    1  853    4    0
##          D    0    0    0  800    2
##          E    0    0    0    0  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.998          
##                  95% CI : (0.9963, 0.999)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9974         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9989   0.9977   0.9950   0.9978
## Specificity            1.0000   0.9992   0.9988   0.9995   1.0000
## Pos Pred Value         1.0000   0.9968   0.9942   0.9975   1.0000
## Neg Pred Value         0.9997   0.9997   0.9995   0.9990   0.9995
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1933   0.1739   0.1631   0.1833
## Detection Prevalence   0.2843   0.1939   0.1750   0.1635   0.1833
## Balanced Accuracy      0.9996   0.9991   0.9982   0.9973   0.9989
```
Accuracy is 0.998 - very good.

#Validation

Write submission files to submit the result to Coursera.

```r
pml_write_files = function(x){
  n = length(x)
  path <- "predictionAssignment_files/answers"
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=file.path(path, filename),quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predict(modFit, validation))
```

All of validation test are correct.
