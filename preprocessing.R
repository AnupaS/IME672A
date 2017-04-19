trn <- read.table("D:/Acads/8th sem/ime672/final/train.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)
tst <- read.table("D:/Acads/8th sem/ime672/final/test.csv", header=TRUE, sep=",", na.strings="NA", dec=".", strip.white=TRUE)

(t <- table(trn$TARGET) / nrow(trn))
#install.packages("plotrix")
require(plotrix)
l <- paste(c('Happy customers\n','Unhappy customers\n'), paste(round(t*100,2), '%', sep=''))
pie3D(t, labels=l, col=c('green','red'), main='Santander customer satisfaction dataset', theta=1, labelcex=0.8)
table(trn$TARGET)/nrow(trn)
summary(trn)
library("Hmisc")
describe(trn)
# remove constant features
for (f in names(trn)) {
  if (length(unique(trn[[f]])) == 1) {
    cat(f, "is constant in train (", unique(trn[[f]]), "). We delete it.\n")
    trn[[f]] <- NULL
    tst[[f]] <- NULL
  }
}

# remove identical features
features_pair <- combn(names(trn), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
  f1 <- pair[1]
  f2 <- pair[2]
  
  if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
    if (all(trn[[f1]] == trn[[f2]])) {
      cat(f1, "and", f2, "are equal.\n")
      toRemove <- c(toRemove, f2)
    }
  }
}

feature.names <- setdiff(names(trn), toRemove)

trn <- trn[, feature.names]
tst <- tst[, feature.names[1:length(feature.names)-1]]


# Removing highly correlated variables
cor_v <- abs(cor(trn))
diag(cor_v) <- 0
cor_v[upper.tri(cor_v)] <- 0
cor_f <- as.data.frame(which(cor_v > 0.85, arr.ind = TRUE))
trn <- trn[,-unique(cor_f$row)]
tst <- tst[,-unique(cor_f$row)]

image(cor_v)

write.csv(trn,"trn.csv")
#install.packages("DMwR")
require(DMwR)

# SMOTE requires the TARGET column to be a factor (not numeric):
trn$TARGET <- factor(trn$TARGET)

# Running SMOTE...
trn.bal <- SMOTE(TARGET ~., trn, perc.over=2427, perc.under=100)
t<-table(trn.bal$TARGET)/nrow(trn.bal)

require(plotrix)
l <- paste(c('Happy customers\n','Unhappy customers\n'), paste(round(t*100,2), '%', sep=''))
pie3D(t, labels=l, col=c('green','red'), main='Santander customer satisfaction dataset', theta=1, labelcex=0.8)

write.csv(trn.bal, "final_train.csv")
write.csv(tst, "final_test.csv")