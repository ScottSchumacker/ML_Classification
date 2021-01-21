# Scott Schumacker
# Multiple Sclerosis Disease Classification From Micro Array Analysis on Auto antibodies

# Loading Libraries
library(caret)
library(glue)
library(dplyr)
library(randomForest)
library(stringr)

###### Data Import and Cleaning #######

# Read in Raw Data Control Data vs. MS Data
CT <- read.csv("/Users/ScottSchumacker/Desktop/MS_CTData - Sheet1.csv", header = F)
MS <- read.csv("/Users/ScottSchumacker/Desktop/MS_MSData - Sheet1.csv", header = F)

# Viewing Raw Data
View(CT)
View(MS)

# Transforming raw data to swap columns with rows
MS_Transform <- t(MS)
View(MS_Transform)

# Changing dataset class to data frame
MS_Transform <- as.data.frame(MS_Transform)
class(MS_Transform)
View(MS_Transform)

# Transforming raw data to swap columns with rows
CT_Transform <- t(CT)
CT_Transform <- as.data.frame(CT_Transform)
class(CT_Transform)
View(CT_Transform)

# Changing first condition column
MS_Transform$V1 <- "Multiple Sclerosis"
MS_Transform[1,1] <- "Condition"

# Adding a header to the dataset
names(MS_Transform) <- MS_Transform[1,]

# Removing duplicate first row
MS_Transform <- MS_Transform[-1,]

# Changing first condition column
CT_Transform$V1 <- "Control"
CT_Transform[1,1] <- "Condition"

# Viewing the data
View(MS_Transform)
View(CT_Transform)

# Checking Dimensions of Both Datasets
dim(MS_Transform)
dim(CT_Transform) 

CT_Transform <- CT_Transform[-c(0,1034:4666)]
View(CT_Transform)

# Adding a header to the dataset
names(CT_Transform) <- CT_Transform[1,]

# Removing duplicate first row
CT_Transform <- CT_Transform[-1,]
View(CT_Transform)

Data <- rbind(MS_Transform, CT_Transform)
dim(Data)
class(Data)
str(Data)

# MS
Data[1:51,1]

#CT
Data[52:80,1]

# Converting all columns to data type numeric
Data[] <- lapply(Data, function(x) as.numeric(as.character(x)))
str(Data)

# Adding back in the Multiple Sclerosis and Control Conditions
Data[1:51,1] <- "Multiple Sclerosis"
View(Data)

Data[52:80,1] <- "Control"
View(Data)
str(Data)
dim(Data)

# Changing condition column to data type factor
Data$Condition <- as.factor(Data$Condition)

# Removing duplicates from the data set
Data <- Data[, !duplicated(colnames(Data))]
dim(Data)
str(Data)

#################### Creating the SVM Classification Model ####################

# Check and remove NA values
sum(is.na(Data))
Data <- na.omit(Data)
sum(is.na(Data))

# Set the seed number
set.seed(100)

# Split the data to training and test
Partition <- createDataPartition(Data$Condition, p=0.7, list = FALSE)
Training_Set <- Data[Partition,]
Testing_Set <- Data[-Partition,]

# Build Training model
Model <- train(Condition ~ ., data = Training_Set,
               method = "svmPoly",
               na.action = na.omit,
               preProcess=c("scale","center"),
               trControl= trainControl(method="none"),
               tuneGrid = data.frame(degree=1,scale=1,C=1)
)

# Build Cross Validation Model
Model.cv <- train(Condition ~ ., data = Training_Set,
                  method = "svmPoly",
                  na.action = na.omit,
                  preProcess=c("scale","center"),
                  trControl= trainControl(method="cv", number=10),
                  tuneGrid = data.frame(degree=1,scale=1,C=1)
)


# Apply model for prediction
Model_Training <-predict(Model, Training_Set)
Model_Testing <-predict(Model, Testing_Set)
Model_Validate <-predict(Model.cv, Training_Set)

# Generate Confusion Matrix
Model_Training_Confusion <-confusionMatrix(Model_Training, Training_Set$Condition)
Model_Testing_Confusion <-confusionMatrix(Model_Testing, Testing_Set$Condition)
Model_Validate_Confusion <-confusionMatrix(Model_Validate, Training_Set$Condition)

# Print the results
print(Model_Training_Confusion)
print(Model_Testing_Confusion)
print(Model_Validate_Confusion)

# Feature importance
Importance <- varImp(Model)
plot(Importance)
plot(Importance, col = "red")

#################### Creating the Random Forest Model ####################

# Set the seed to replicate the results
set.seed(100)

# Cleaning the Column Names to remove underscores and dashes
ColumnNames <- colnames(Data)
View(ColumnNames)
ColumnNames <- str_remove_all(ColumnNames, "_")
ColumnNames <- str_remove_all(ColumnNames, "-")
ColumnNames
colnames(Data) <- ColumnNames
View(Data)
str(Data)

# Creating Random Forest Model
RFModel <- randomForest(Condition ~., data = Data, proximity = TRUE)

# Viewing Model Performance
RFModel

library(ggplot2)

oob.error.data <- data.frame(
  Trees=rep(1:nrow(RFModel$err.rate), times=3),
  type=rep(c("OOB", "Control", "Multiple Sclerosis"), each=nrow(RFModel$err.rate)),
  
)