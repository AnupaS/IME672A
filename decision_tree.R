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

#install.packages("tree")
require('tree')
tr <- tree(TARGET ~., train, method = 'class')
test.y <- test$TARGET
plot(tr)
text(tr)
require("ROCR")
pred <- predict(tr,newdata = test)
pred2 <- prediction(pred,  (test.y))
roc = performance(pred2, measure = "tpr", x.measure = "fpr")
plot(roc)
abline(a=0,b=1)

auc <- performance(pred2, 'auc')
#install.packages("caret")
require(caret)

confusionMatrix(as.numeric(pred>0.5), test.y)

predtest <- predict(tr, tst)

tst$target <- as.numeric(predtest>0.5)
tst$target
write.csv(tst, "D:/Acads/8th sem/ime672/final/predictedfile_decisiontree.csv")







