library(caret)
train <- read.csv("german_credit-withLabels.csv")
test <- read.csv("german_credit_test.csv")
summary(train)
summary(test)

myvars <- c("Creditability","Account.Balance", "Payment.Status.of.Previous.Credit", "Most.valuable.available.asset",
            "Purpose", "Length.of.current.employment", "Credit.Amount", "Value.Savings.Stocks",
            "Duration.of.Credit..month.","Type.of.apartment")
new_train <- train[myvars]
new_test <- test[myvars]

new_train$Creditability[new_train$Creditability == "0"] <- "No"
new_train$Creditability[new_train$Creditability == "1"] <- "Yes"
new_test$Creditability[new_test$Creditability == "0"] <- "No"
new_test$Creditability[new_test$Creditability == "1"] <- "Yes"

new_train$Creditability <- as.factor(new_train$Creditability)
new_train$Account.Balance <- as.factor(new_train$Account.Balance)
new_train$Payment.Status.of.Previous.Credit <- as.factor(new_train$Payment.Status.of.Previous.Credit)
new_train$Most.valuable.available.asset <- as.factor(new_train$Most.valuable.available.asset)
new_train$Purpose <- as.factor(new_train$Purpose)
new_train$Length.of.current.employment <- as.factor(new_train$Length.of.current.employment)
new_train$Value.Savings.Stocks <- as.factor(new_train$Value.Savings.Stocks)
new_train$Type.of.apartment <- as.factor(new_train$Type.of.apartment)

new_test$Creditability <- as.factor(new_test$Creditability)
new_test$Account.Balance <- as.factor(new_test$Account.Balance)
new_test$Payment.Status.of.Previous.Credit <- as.factor(new_test$Payment.Status.of.Previous.Credit)
new_test$Most.valuable.available.asset <- as.factor(new_test$Most.valuable.available.asset)
new_test$Purpose <- as.factor(new_test$Purpose)
new_test$Length.of.current.employment <- as.factor(new_test$Length.of.current.employment)
new_test$Value.Savings.Stocks <- as.factor(new_test$Value.Savings.Stocks)
new_test$Type.of.apartment <- as.factor(new_test$Type.of.apartment)


split=0.70
trainIndex <- createDataPartition(new_train$Creditability, p=split, list=FALSE)
data_train <- new_train[ trainIndex,]
data_validation <- new_train[-trainIndex,]

fitControl <- trainControl(
  method = "none",
  savePredictions = TRUE,
  classProbs = TRUE,
  summaryFunction = multiClassSummary)

fitControl2 <- trainControl(## 10-fold Crossvalidation
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 3,
  summaryFunction = multiClassSummary)

set.seed(3456)
dt <- caret::train(Creditability ~ ., data = data_train, method = "rpart2",
                   trControl = fitControl,
                   tuneGrid = data.frame(maxdepth=4))
# 
# set.seed(3456)
# dt2 <- caret::train(Creditability ~ ., data = data_train, method = "rpart2",
#                    trControl = fitControl2)

# library(rpart.plot)
# rpart.plot(dt2$finalModel)

x_data_train = subset(data_train, select = -c(Creditability) )
dt_train_predict = predict(dt, x_data_train)
confusionMatrix(dt_train_predict, data_train$Creditability)

x_data_validation = subset(data_validation, select = -c(Creditability) )
dt_valid_predict = predict(dt, x_data_validation)
confusionMatrix(dt_valid_predict, data_validation$Creditability)

x_new_test = subset(new_test, select = -c(Creditability) )
dt_test_predict = predict(dt, x_new_test)
confusionMatrix(dt_test_predict, new_test$Creditability)


#maxdepth=8
set.seed(3456)
dt2 <- caret::train(Creditability ~ ., data = data_train, method = "rpart2",
                   trControl = fitControl,
                   tuneGrid = data.frame(maxdepth=8))


x_data_train2 = subset(data_train, select = -c(Creditability) )
dt_train_predict2 = predict(dt2, x_data_train)
confusionMatrix(dt_train_predict2, data_train$Creditability)

dt_valid_predict2 = predict(dt2, x_data_validation)
confusionMatrix(dt_valid_predict2, data_validation$Creditability)

dt_test_predict2 = predict(dt2, x_new_test)
confusionMatrix(dt_test_predict2, new_test$Creditability)
dt2_test_predict_prob = predict(dt2, x_new_test, type="prob")

#knn
set.seed(3456)
knn <- caret::train(Creditability ~ ., data = new_train, method = "knn",
                   trControl = fitControl,
                   tuneGrid = data.frame(k=7), preProcess = c("center","scale"))

# dt2 <- caret::train(Creditability ~ ., data = new_train, method = "knn",
#                     trControl = fitControl2, preProcess = c("center","scale"))


x_new_train = subset(new_train, select = -c(Creditability))
knn_train_predict = predict(knn, x_new_train)
confusionMatrix(knn_train_predict, new_train$Creditability)


knn_test_predict = predict(knn, x_new_test)
confusionMatrix(knn_test_predict, new_test$Creditability)
knn_test_predict_prob = predict(knn, x_new_test, type="prob")

#k=4
set.seed(3456)
knn2 <- caret::train(Creditability ~ ., data = new_train, method = "knn",
                    trControl = fitControl,
                    tuneGrid = data.frame(k=4), preProcess = c("center","scale"))

# dt2 <- caret::train(Creditability ~ ., data = new_train, method = "knn",
#                     trControl = fitControl2)


knn_train_predict2 = predict(knn2, x_new_train)
confusionMatrix(knn_train_predict2, new_train$Creditability)


knn_test_predict2 = predict(knn2, x_new_test)
confusionMatrix(knn_test_predict2, new_test$Creditability)


#naive bayes
set.seed(3456)
nb <- caret::train(Creditability ~ ., data = new_train, method = "nb",
                    trControl = fitControl,
                    tuneGrid = data.frame(fL = 0, usekernel = FALSE,adjust = 1), preProcess = c("center","scale"))

# dt2 <- caret::train(x_new_train,new_train$Creditability, method = "nb",
#                     trControl = fitControl2,preProcess = c("center","scale"))

nb_train_predict = predict(nb, x_new_train)
confusionMatrix(nb_train_predict, new_train$Creditability)


nb_test_predict = predict(nb, x_new_test)
confusionMatrix(nb_test_predict, new_test$Creditability)
nb_test_predict_prob = predict(nb, x_new_test, type="prob")

#use kernel
set.seed(3456)
nb2 <- caret::train(Creditability ~ ., data = new_train, method = "nb",
                   trControl = fitControl,
                   tuneGrid = data.frame(fL = 0, usekernel = TRUE,adjust = 1), preProcess = c("center","scale"))

nb2_train_predict = predict(nb2, x_new_train)
confusionMatrix(nb2_train_predict, new_train$Creditability)


nb2_test_predict = predict(nb2, x_new_test)
confusionMatrix(nb2_test_predict, new_test$Creditability)

#neural network
set.seed(3456)
nn <- caret::train(Creditability ~ ., data = new_train, method = "nnet",
                    trControl = fitControl,
                    tuneGrid = data.frame(size=1, decay=0.1), preProcess = c("center","scale"))

# dt2 <- caret::train(Creditability ~ ., data = new_train, method = "nnet",
#                     trControl = fitControl2, preProcess = c("center","scale"))


nn_train_predict = predict(nn, x_new_train)
confusionMatrix(nn_train_predict, new_train$Creditability)


nn_test_predict = predict(nn, x_new_test)
confusionMatrix(nn_test_predict, new_test$Creditability)


#size=2
set.seed(3456)
nn2 <- caret::train(Creditability ~ ., data = new_train, method = "nnet",
                     trControl = fitControl,
                     tuneGrid = data.frame(size=2, decay=0.1), preProcess = c("center","scale"))

# dt2 <- caret::train(Creditability ~ ., data = new_train, method = "knn",
#                     trControl = fitControl2)


nn_train_predict2 = predict(nn2, x_new_train)
confusionMatrix(knn_train_predict2, new_train$Creditability)


nn_test_predict2 = predict(nn2, x_new_test)
confusionMatrix(nn_test_predict2, new_test$Creditability)
nn2_test_predict_prob = predict(nn2, x_new_test, type="prob")

#svm
set.seed(3456)
svm <- caret::train(Creditability ~ ., data = new_train, method = "svmLinear",
                   trControl = fitControl,
                   tuneGrid = data.frame(C=1), preProcess = c("center","scale"))

# dt2 <- caret::train(Creditability ~ ., data = new_train, method = "svm",
#                     trControl = fitControl2, preProcess = c("center","scale"))


svm_train_predict = predict(svm, x_new_train)
confusionMatrix(svm_train_predict, new_train$Creditability)


svm_test_predict = predict(svm, x_new_test)
confusionMatrix(svm_test_predict, new_test$Creditability)
svm_test_predict_prob = predict(svm, x_new_test, type="prob")

#C=2
set.seed(3456)
svm2 <- caret::train(Creditability ~ ., data = new_train, method = "svmLinear",
                    trControl = fitControl,
                    tuneGrid = data.frame(C=100), preProcess = c("center","scale"))

# dt2 <- caret::train(Creditability ~ ., data = new_train, method = "knn",
#                     trControl = fitControl2)


svm_train_predict2 = predict(svm2, x_new_train)
confusionMatrix(knn_train_predict2, new_train$Creditability)


svm_test_predict2 = predict(svm2, x_new_test)
confusionMatrix(svm_test_predict2, new_test$Creditability)

#logistic regression
set.seed(3456)
lr <- caret::train(Creditability ~ ., data = new_train, method = "glm",
                    trControl = fitControl,
                    preProcess = c("center","scale"), family=binomial)

# dt2 <- caret::train(Creditability ~ ., data = new_train, method = "glm",
#                     trControl = fitControl2, preProcess = c("center","scale"),family=binomial)


lr_train_predict = predict(lr, x_new_train)
confusionMatrix(lr_train_predict, new_train$Creditability)


lr_test_predict = predict(lr, x_new_test)
confusionMatrix(lr_test_predict, new_test$Creditability)

lr_test_predict_prob = predict(lr, x_new_test, type="prob")

x_new_test$pred_weighted_avg<-((dt2_test_predict_prob$Yes)+(knn_test_predict_prob$Yes)+
                                                                   (nb_test_predict_prob$Yes)+
                                                                      (nn2_test_predict_prob$Yes)+
                                                                         (svm_test_predict_prob$Yes)+
                                                                            (lr_test_predict_prob$Yes))/6
x_new_test$pred_weighted_avg<-as.factor(ifelse(x_new_test$pred_weighted_avg<0.5,'No','Yes'))
confusionMatrix(x_new_test$pred_weighted_avg, new_test$Creditability)

VarImportanceDT2 <- varImp(dt2)
plot(VarImportanceDT2, main = "Predictor Importance")

VarImportanceKNN <- varImp(knn)
plot(VarImportanceKNN, main = "Predictor Importance")

VarImportanceNB <- varImp(nb)
plot(VarImportanceNB, main = "Predictor Importance")

VarImportanceNN2 <- varImp(nn2)
plot(VarImportanceNN2, main = "Predictor Importance")

VarImportanceSVM <- varImp(svm)
plot(VarImportanceSVM, main = "Predictor Importance")

VarImportanceLR <- varImp(lr)
plot(VarImportanceLR, main = "Predictor Importance")
