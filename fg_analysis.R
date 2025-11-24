# =============================================
# ----- Field Goal Decision Trees -----
# ============================================= 

# --- Document Set Up
library(dplyr)
library(lubridate)
library(randomForest)
library(ggplot2)
library(pROC)
library(rpart) 
library(rpart.plot)
rm(list = ls())

# --- Read in the data
fg_data = read.csv("nfl_fg_data_2001_2024.csv")
head(fg_data)

# --------- Exploratory Analysis ---------

# Bar chart of FG made vs missed
fg_data$fg_made <- as.character(fg_data$fg_made)
ggplot(data = fg_data) + 
  geom_bar(aes(x= fg_made))

# Histogram of kick distance
ggplot(data = fg_data) + 
  geom_histogram(aes(x= kick_distance))

# Histogram of wind speed
ggplot(data = fg_data) +
  geom_histogram(aes(x= wind))

# --------- Multivariate Plots -----------
# We see from our exploratory analysis significantly more field goals 
# were made than missed, but what factors influenced the probablity of a make


# --------- Data Preparation for Decision Trees ---------

# Set our seed so we all get the same random numbers
RNGkind(sample.kind = "default")
set.seed(172)

#Create a vector of randomly selected rows that will go into training
train.idx <- sample(x=1:nrow(fg_data), size = 0.8*nrow(fg_data))
train.df <- fg_data[train.idx,]
test.df <- fg_data[-train.idx,] 

# --------- Tree Fitting ---------
# Set our seed so we all get the same random numbers
RNGkind(sample.kind = "default")
set.seed(172)

#To start, I will make a ginormous tree and tune
tree <- rpart(fg_made ~ ., 
              data = train.df, method = 'class',
              control = rpart.control(cp=0.0001,minsplit = 1))
printcp(tree)

# In order to minimize xerror, the best number of splits is 2 

# To automate our process and ensuring the best tree is always generated,
# we will use the following code and make a new, better tree

# Set our seed so we all get the same random numbers
RNGkind(sample.kind = "default")
set.seed(172)

optimalcp <-  tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"]
best_tree <- prune(tree, optimalcp)
rpart.plot(best_tree)


###=== MODEL VALIDATION ---
#need to make a column of predictions
#need to do that on the testing data 
test.df$fg_pred <- predict(best_tree, test.df, type = "class")
head(test.df)

#make a confusion matrix 
table(test.df$fg_pred, test.df$fg_made)
# Accuracy: (42+4070) / (42+4070+33+793) = 0.832

pi_hat <- predict(best_tree, test.df, type = "prob")[,"1"]

#How well do we predict with the model on test data?
#if cutoff p = .1
p <- 0.10
y_hat <- as.factor(ifelse(pi_hat > p, "1","0"))
# Make new confusion matrix based on pi* = 0.1
table(y_hat,test.df$fg_made)

# if cutoff p = 0.5
p <- 0.50
y_hat <- as.factor(ifelse(pi_hat > p, "1","0"))
# Make new confusion matrix based on pi* = 0.5
table(y_hat,test.df$fg_made)
# Sensitivity : 
# Specificity :

# If cutoff p = 0.9
p <- 0.9
y_hat <- as.factor(ifelse(pi_hat > p, "1","0"))
# Make new confusion matrix based on pi* = 0.9
table(y_hat,test.df$fg_made)

# ======= ROC Curve ==========
rocCurve <- roc(response = test.df$fg_made,
                predictor = pi_hat,
                levels = c("0","1"))

plot(rocCurve, print.thres = "all", print.auc = TRUE)

# If we set pi* = 0.819, we can achieve a specificity of 0.734 and sensitivity of 0.649

# That is, we will predict a miss 73.4% of the time when there is a miss.
# WE will predict a make 64.9% of the time when there is actually a make.
# Area under the curve is 0.699


