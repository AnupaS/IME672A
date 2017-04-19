trn.bal <- read.table("D:/Acads/8th sem/ime672/final/final_train.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
tst <- read.table("D:/Acads/8th sem/ime672/final/final_test.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
smp_size <- floor(0.75 * nrow(trn.bal))
set.seed(1)
train_ind <- sample(seq_len(nrow(trn.bal)), size = smp_size)
train <- trn.bal[train_ind, ]
test <- trn.bal[-train_ind, ]
train.y <- train$TARGET
test.y <- test$TARGET
#train$TARGET <- NULL
#install.packages("randomForest")
library(randomForest)
set.seed(1234)
library(caret)
fit <- randomForest(train.y ~ ., y = train$TARGET, data=train, importance=TRUE,ntree=8)
#install.packages("ROCR")
require(ROCR)
predictions = predict(fit,test)
summary(predictions)
pred2 <- prediction(predictions,  (test.y))
(auc <- performance(pred2, 'auc'))
roc = performance(pred2, measure = "tpr", x.measure = "fpr")
plot(roc)
abline(a=0,b=1)

pred <- ifelse(predictions>0.5,1,0)
library(caret)
library(e1071)
confusionMatrix(data=factor(pred),reference=factor(test.y),positive='1')
importance(fit)
varImpPlot(fit)
plot(fit)
pred1 <- predict(fit,tst)