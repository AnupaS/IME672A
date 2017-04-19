trn.bal <- read.table("D:/Acads/8th sem/ime672/final/final_train.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
tst <- read.table("D:/Acads/8th sem/ime672/final/final_test.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
trn <- read.table("D:/Acads/8th sem/ime672/final/trn.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)

trn$X<-NULL
trn.bal$X<-NULL
tst$X<-NULL

smp_size <- floor(0.75 * nrow(trn.bal))
set.seed(1)
train_ind <- sample(seq_len(nrow(trn.bal)), size = smp_size)
train <- trn.bal[train_ind, ]
#test <- trn.bal[-train_ind, ]
smp_size <- floor(0.5 * nrow(trn))
set.seed(1)
train_ind <- sample(seq_len(nrow(trn)), size = smp_size)
test <- trn[-train_ind, ]


trn.bal.subset <- train[sample(nrow(train), 50000),]


trn.bal.subset$TARGET <- factor(trn.bal.subset$TARGET)

require(e1071)

# fit an SVM
model <- svm(TARGET ~., trn.bal.subset)

# do prediction
pred <- predict(model, test[,1:ncol(test)-1])

require(ROCR)
pred2 <- prediction(as.numeric(pred), test$TARGET)
(auc <- performance(pred2, 'auc'))

roc = performance(pred2, measure = "tpr", x.measure = "fpr")
plot(roc)
abline(a=0,b=1)

require(caret)
confusionMatrix(pred, test$TARGET)
predtest <- predict(model, tst, type = "response")

tst$target <- predtest
write.csv(tst, "D:/Acads/8th sem/ime672/final/predictedfile_svm.csv")
