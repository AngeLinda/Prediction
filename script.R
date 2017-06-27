rm(list=ls())

## prepare library
library(caret)
library(rpart)
library(rattle)
library(randomForest)
library(class)
library(e1071)

## Load & Read the data
training_Url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing_Url <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(training_Url, na.strings = c("NA","#DIV/0!",""))
testing <- read.csv(testing_Url, na.strings = c("NA","#DIV/0!",""))

## Clean the data
training_set <- training[, colSums(is.na(training)) == 0]
training_set <- training_set[, -c(1:7)]
testing_set <- testing[, colSums(is.na(testing)) == 0]
testing_set <- testing_set[, -c(1:7)]
testing_set <- testing_set[, -53]

## Partition the training set into training data and validation data
inTrain <- createDataPartition(training_set$classe, p=0.75, list = F)
training_data <- training_set[inTrain,]
validation_data <- training_set[-inTrain,]

## prediction model 1: Decision Tree
model_DT <- rpart(classe ~ ., method = "class",data = training_data)
pred_DT <- predict(model_DT, newdata = validation_data, type="class")
confusionMatrix(pred_DT, validation_data$classe)

fancyRpartPlot(model_DT)

## prediction model 2: Random Forest
model_RF <- randomForest(classe ~., data = training_data) 
pred_RF <- predict(model_RF, validation_data)
confusionMatrix(pred_RF, validation_data$classe)

plot(model_RF)

## prediction model 3: k-NN
cl <- training_data$classe
acc <- NULL
for(i in 1:20) {
  model_KNN <- knn(training_data[, -53], validation_data[,-53], cl, k=i, prob=TRUE) 
  acc <- c(acc,confusionMatrix(model_KNN, validation_data$classe)$overall[[1]])
}
plot(acc*100, xlab = "k value", ylab = "accuracy [%]",
     main = "accuracy result with increase of K value")

## prediction model 4: SVM
model_SVM <- svm(classe ~., data = training_data)
pred_SVM <- predict(model_SVM, validation_data)
confusionMatrix(pred_SVM, validation_data$classe)

## table for comparison
tb <- matrix(c(confusionMatrix(pred_DT, validation_data$classe)$overall[[1]],
               confusionMatrix(pred_RF, validation_data$classe)$overall[[1]],
               max(acc),
               confusionMatrix(pred_SVM, validation_data$classe)$overall[[1]]), ncol=1, byrow = T)
colnames(tb) <- c("Accuracy")
rownames(tb) <- c("Decision Tree", "Random Forest", 
                  "K Nearest Neighbor (K=1), Support Vector Machines")
tb <- as.table(tb)

## final predictor: random forest
pred_final <- predict(model_RF, testing_set)




