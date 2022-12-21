## Importing Library
library(xgboost)
library(modeldata)
library(dplyr)
library(caret)
library(corrplot)

## Importing Data
data <- read.csv("final_data.csv",sep = ",",header = T, na.strings = "?")


## data Manipulation

x <- data %>% select(Depression, HTN,	Smoking,	F_History,	Diabetes,	BP,	hemoglobin,	platelete_count,	cholestrol,	Diagnosis) 

unlist(x)

daf <- cor(x)

corrplot(daf, order = "AOE", method = "color", addCoef.col = "black")

summary(x)

is.null(x)

unique.data.frame(x)


## data splitting

set.seed(1234)

ind <- sample(2, nrow(x), replace = T, prob = c(0.7, 0.3))
train <- x[ind==1,]
test <- x[ind==2,]


## matrix creation
train_x = data.matrix(train[, -1])   
train_y = train[,1]

test_x = data.matrix(test[, -1])
test_y = test[, 1]

## final matrix
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

#define watchlist
watchlist = list(train=xgb_train, test=xgb_test)


## setting parameter
param <-  list(set.seed = 1500, 
               eval_mertic = "mlogloss", 
               objective = "binary:logistic")

## XGboost
model <- xgb.train(data = xgb_train,
                   params = param,
                   nrounds = 100,
                   watchlist = watchlist,
                   eta = 0.01)

## testing model
xgb.plot.shap(data = train_x,
              model = model,
              top_n =5)

e <- data.frame(model$evaluation_log)
plot(e$iter, e$train_logloss,  col = 'blue')
lines(e$iter, e$test_logloss, col = "red")




pred_y = predict(model, xgb_test)

mean((test_y - pred_y)^2) #mse
caret::MAE(test_y, pred_y) #mae
caret::RMSE(test_y, pred_y) #rmse


## importance
imp <- xgb.importance(colnames(xgb_train),
                      model = model)

xgb.plot.importance(imp)


## confusion matrix

xgbpred <- predict(model,xgb_test)
xgbpred <- ifelse(xgbpred > 0.5,1,0)


confusionMatrix (as.factor(xgbpred), as.factor(test_y))
