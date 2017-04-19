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

#install.packages("gbm")
require(gbm)
gbmFit.ada = gbm(formula = train$TARGET ~.,
                 distribution = 'adaboost',
                 data = train,
                 n.trees = 5,        #the number of trees in the model
                 interaction.depth = 4,  #each tree will evaluate 4 decisions
                 n.minobsinnode = 2,     #higher means more conservative fit
                 shrinkage = .01,        #the learning rate
                 bag.fraction = 1,     #subsampling fraction
                 train.fraction = 1,   #fraction of data for training
                 cv.folds = 5)           #running five-fold cross-validation

pred <- predict(gbmFit.ada, test, n.trees = 5, type = 'response')
require(ROCR)
pred2 <- prediction(pred, test$TARGET)
(auc <- performance(pred2, 'auc'))
roc = performance(pred2, measure = "tpr", x.measure = "fpr")
plot(roc)
abline(a=0,b=1)
require(caret)
confusionMatrix(as.numeric(pred>0.5), test$TARGET)
predtest <- predict(gbmFit.ada, tst, type = "response")
tst$target <- as.numeric(predtest>0.5)
write.csv(tst, "D:/Acads/8th sem/ime672/final/predictedfile_adaboost.csv")
