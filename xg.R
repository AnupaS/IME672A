trn.bal <- read.table("D:/Acads/8th sem/ime672/final/final_train.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
tst <- read.table("D:/Acads/8th sem/ime672/final/final_test.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
trn <- read.table("D:/Acads/8th sem/ime672/final/trn.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
smp_size <- floor(0.75 * nrow(trn.bal))
set.seed(1)
train_ind <- sample(seq_len(nrow(trn.bal)), size = smp_size)
train <- trn.bal[train_ind, ]

smp_size <- floor(0.5 * nrow(trn))
set.seed(1)
train_ind <- sample(seq_len(nrow(trn)), size = smp_size)
test <- trn[-train_ind, ]

##XGBoost
# extract the TARGET values
train.y <- train$TARGET
test.y <- test$TARGET
trn.bal.y <- trn.bal$TARGET

# remove the TARGET columns

train$TARGET <- NULL
test$TARGET <- NULL

# limit vars in test based on min and max vals of train
print('Setting min-max lims on test data')
for(f in colnames(train))
{
  lim <- min(train[,f])
  test[test[,f] < lim, f] <- lim
  
  lim <- max(train[,f])
  test[test[,f] > lim, f] <- lim 
}

# restore the TARGET values
train$TARGET <- train.y
test$TARGET <- test.y
#install.packages("xgboost")
library(xgboost)
library(Matrix)
set.seed(1234)
trn.bal1 <- sparse.model.matrix(TARGET ~ ., data = trn.bal)
dtrn.bal <- xgb.DMatrix(as.matrix(trn.bal), label=trn.bal.y)
watchlist <- list(train=dtrn.bal)

param <- list( objective        = "binary:logistic",
               booster          = "gbtree",
               eval_metric      = "auc",
               eta              = 0.01,
               max_depth        = 7,
               subsample        = 0.8,
               colsample_bytree = 0.8
)


clf <- xgb.train(params       = param,
                 data      = dtrn.bal,
                 nrounds   = 1,
                 verbose   = 1,
                 watchlist = watchlist,
                 maximize  = FALSE
)
test$TARGET <- NULL
# evaluate
pred <- predict(clf, data.matrix(test))
require(ROCR)
pred2 <- prediction(pred, test.y)
(auc <- performance(pred2, 'auc'))
roc = performance(pred2, measure = "tpr", x.measure = "fpr")
plot(roc)
abline(a=0,b=1)
require(caret)
confusionMatrix(as.numeric(pred>0.5), test.y)
predtest <- predict(clf, data.matrix(tst))

tst$TARGET <- as.numeric(predtest>0.5)
write.csv(tst, "D:/Acads/8th sem/ime672/final/predictedfile_xgboost.csv")