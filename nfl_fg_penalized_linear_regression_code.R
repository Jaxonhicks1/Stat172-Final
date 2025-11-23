# ===============================================================
# ---------- NFL FG Data - Penalized Linear Regression ----------
# ===============================================================

# Our goal is to build a parsimonious GLM to understand important relationships between 
# our predictor variables and whether a field goal is made. To do this, we can use 
# penalized linear regression, specifically Lasso and Ridge. While this provides a 
# descriptive model to describe relationships, we can also use penalized linear 
# regression to create a prediction tool by splitting data into a train and test split. 



# ======================================================================
# ---------- 0) Prep (load libraries, data exploration, etc.) ----------
# ======================================================================

# clear environment
rm(list = ls())

# load libraries
library(tidyverse)
library(pROC)
library(glmnet)
library(RColorBrewer)

# load data
fg_data <- read.csv("nfl_fg_data_2001_2024.csv")

# view data
glimpse(fg_data)
view(fg_data)
summary(fg_data)
str(fg_data)

# specify categorical variables as factors
fg_data <- fg_data %>%
  mutate(
    down = as.factor(down),
    qtr = as.factor(qtr),
    home = as.factor(home),
    indoor = as.factor(indoor),
    precipitation = as.factor(precipitation),
    grass = as.factor(grass)
  )

# verify  
str(fg_data)


# =============================================
# ---------- 1) Train and Test Split ----------
# =============================================

# Since one of our goals is prediction, we need to split our data into a train/test split

RNGkind(sample.kind = "default")
set.seed(172)

train.idx <- sample(x = 1:nrow(fg_data), size = .8*nrow(fg_data))
train.df <- fg_data[train.idx,]
test.df <- fg_data[-train.idx,]



# ========================================================
# ---------- 2) Traditional Logistic Regression ----------
# ========================================================

# we are starting off with traditional logistic regression so we can compare it
# to the penalized methods. 

lr_mle <- glm(fg_made ~ . ,
              data = train.df,
              family = binomial(link = "logit"))

lr_mle_coefs <- coef(lr_mle)



# ===================================================
# ---------- 3) Lasso and Ridge Regression ----------
# ===================================================

# Create matrix which one-hot codes all factor variables
x.train <- model.matrix(fg_made ~ ., data = train.df)[,-1]
x.test <- model.matrix(fg_made ~ ., data = test.df)[,-1]

# Create vectors of 0/1 y variable
y.train <- (train.df$fg_made) %>% as.vector
y.test <- as.vector(test.df$fg_made)

# Use cross validation and fit a series of lasso and ridge models
# and calculate how well they do out of sample for each lambda value
lr_lasso_cv <- cv.glmnet(x.train, 
                         y.train, 
                         family = binomial(link = "logit"), 
                         alpha = 1)  # sets us in lasso mode

lr_ridge_cv <- cv.glmnet(x.train, 
                         y.train, 
                         family = binomial(link = "logit"), 
                         alpha = 0)  # sets us in ridge mode


# ----- Investigating Results -----
# ---------------------------------

# Plot results from cross validation procedures
plot(lr_lasso_cv, sign.lambda = 1)
plot(lr_ridge_cv, sign.lambda = 1)

# For the above plots: 
#   - right side = big lambda = intercept-only model (hard shrinkage)
#   - left side = small lambda = MLE (no shrinkage)
# Based on these graphs (which we are using to visualize shrinkage), lasso and ridge
# did not appear to shrink the number of variables very much, if at all. 
# The best lambdas saled below will likely be close to 0. 

# saving best lambdas
best_lasso_lambda <- lr_lasso_cv$lambda.min
best_ridge_lambda <- lr_ridge_cv$lambda.min

best_lasso_lambda
best_ridge_lambda

# Saving Ridge and Lasso coefficients
lr_ridge_coefs <- coef(lr_ridge_cv, s = "lambda.min") %>% as.matrix()
lr_lasso_coefs <- coef(lr_lasso_cv, s = "lambda.min") %>% as.matrix()

lr_ridge_coefs
lr_lasso_coefs

# Comparing coefficients from penalized regression to standard logistic regression
# Ridge vs MLE
ggplot() + 
  geom_point(aes(x = lr_mle_coefs, y = lr_ridge_coefs)) + 
  geom_abline(aes(intercept = 0, slope = 1)) + 
  xlim(c(-10,10)) + ylim(c(-10,10))

# Lasso vs MLE
ggplot() + 
  geom_point(aes(x = lr_mle_coefs, y = lr_lasso_coefs)) + 
  geom_abline(aes(intercept = 0, slope = 1)) + 
  xlim(c(-10,10)) + ylim(c(-10,10))


# ----- Final Lasso and Ridge Models -----
# ----------------------------------------

# final Lasso model
final_lasso <- glmnet(x.train, y.train, 
                      family = binomial(link = "logit"),
                      alpha = 1,
                      lambda = best_lasso_lambda)

# final Ridge Model
final_ridge <- glmnet(x.train, y.train, 
                      family = binomial(link = "logit"),
                      alpha = 0,
                      lambda = best_ridge_lambda)



# ============================================
# ---------- Prediction Performance ----------
# ============================================

# Calculating predictions on test data
test.df.preds <- test.df %>% 
  mutate(mle_pred = predict(lr_mle, test.df, type = "response"),
         lasso_pred = predict(final_lasso, x.test, type = "response")[,1],
         ridge_pred = predict(final_ridge, x.test, type = "response")[,1])

# Comparing MLE vs Lasso and ridge predictions
cor(test.df.preds$mle_pred, test.df.preds$lasso_pred)
cor(test.df.preds$mle_pred, test.df.preds$ridge_pred)
plot(test.df.preds$mle_pred, test.df.preds$lasso_pred)
plot(test.df.preds$mle_pred, test.df.preds$ridge_pred)

# Createing ROC Curves
mle_rocCurve <- roc(response = as.factor(test.df.preds$fg_made),
                    predictor = test.df.preds$mle_pred,
                    levels = c("0", "1"))

lasso_rocCurve <- roc(response = as.factor(test.df.preds$fg_made),
                      predictor = test.df.preds$lasso_pred,
                      levels = c("0", "1"))

ridge_rocCurve <- roc(response = as.factor(test.df.preds$fg_made),
                      predictor = test.df.preds$ridge_pred,
                      levels = c("0", "1"))

# Plotting ROC curves
plot(mle_rocCurve,print.thres = TRUE, print.auc = TRUE)
plot(lasso_rocCurve,print.thres = TRUE, print.auc = TRUE)
plot(ridge_rocCurve,print.thres = TRUE, print.auc = TRUE)

# ----- Creating Combined ROC Curve --------
# ------------------------------------------

# Make data frame of MLE ROC info
mle_data <- data.frame(
  Model = "MLE",
  Specificity = mle_rocCurve$specificities,
  Sensitivity = mle_rocCurve$sensitivities,
  AUC = as.numeric(mle_rocCurve$auc)
)

# Make data frame of lasso ROC info
lasso_data <- data.frame(
  Model = "Lasso",
  Specificity = lasso_rocCurve$specificities,
  Sensitivity = lasso_rocCurve$sensitivities,
  AUC = lasso_rocCurve$auc %>% as.numeric
)

# Make data frame of ridge ROC info
ridge_data <- data.frame(
  Model = "Ridge",
  Specificity = ridge_rocCurve$specificities,
  Sensitivity = ridge_rocCurve$sensitivities,
  AUC = ridge_rocCurve$auc%>% as.numeric
)

# Combine all the data frames
roc_data <- rbind(mle_data, lasso_data, ridge_data)

# Plot comparing standard MLE to Lasso and Ridge
ggplot() +
  geom_line(aes(x = 1 - Specificity, y = Sensitivity, color = Model),data = roc_data) +
  geom_text(data = roc_data %>% group_by(Model) %>% slice(1), 
            aes(x = 0.75, y = c(0.75, 0.65, 0.55), colour = Model,
                label = paste0(Model, " AUC = ", round(AUC, 3)))) +
  scale_colour_brewer(palette = "Paired") +
  labs(x = "1 - Specificity", y = "Sensitivity", color = "Model") +
  theme_minimal()

# The above graph demonstrates the the two penalized linear regression methods used,
# Lasso and Ridge, produced very similar models to the standard MLE logistic regression
# model, which is in line with the earlier analysis above (such as very small lambdas).



# =======================================================
# ---------- Descriptive Model Interpretations ----------
# =======================================================

# Finding the best model (model with lowest AUC)
as.numeric(lasso_rocCurve$auc)
as.numeric(ridge_rocCurve$auc)
as.numeric(mle_rocCurve$auc)
# The model with the highest AUC is the Lasso regression model

# lasso curve
plot(lasso_rocCurve,print.thres = TRUE, print.auc = TRUE)

# Lasso coefficients
lr_lasso_coefs


# ----- LASSO Coefficients -----
# ------------------------------
# Intercept: 5.250897
# kick_distance: -0.1008369
# down2: 0.1546646
# down3: 0.00
# down4: 0.2822978
# ydstogo: -0.0008847196
# qtr4: 0.00            
# qtr5: 0.00
# qtr4: 0.04431652
# qtr5: -0.2611250
# qtr6: 1.197312
# game_seconds_remaining: -0.000003948070
# home1: 0.03430091
# indoor1: 0.02050197
# temp: 0.005811582
# wind: -0.01723645
# precipitation1: -0.2452419
# grass1: -0.1075057