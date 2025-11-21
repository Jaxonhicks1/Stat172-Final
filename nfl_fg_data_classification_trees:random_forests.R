# =======================================
# ------ NFL FG Data Random Forest ------
# =======================================

# ================================================
# ----- 0) Clear Environment & Load Packages ----- 
# ================================================

rm(list = ls())

library(randomForest)
library(ggplot2) 
library(pROC) 
library(tidymodels) 

# ===========================
# ----- 1) Load FG Data ----- 
# ===========================

fg_data <- read.csv("/Users/josephjoswiak/Desktop/Fall 2025 Classes/STAT 172/nfl_fg_data_2001_2024.csv")

# ===========================
# ----- 2) Set the seed ----- 
# ===========================

RNGkind(sample.kind = "default")
set.seed(172)

# =========================================
# ----- 3) Make training/testing data ----- 
# =========================================
#Converted train.df and test.df to categorical classes for classification.
#Random Forest treats numeric 0/1 as regression unless converted
#train.df$fg_made <- factor(train.df$fg_made)
#test.df$fg_made <- factor(test.df$fg_made)

train.idx <- sample(x = 1:nrow(fg_data), size = floor(.8*nrow(fg_data)))

train.df <- fg_data[train.idx,]

test.df <- fg_data[-train.idx,]

train.df$fg_made <- factor(train.df$fg_made)

test.df$fg_made <- factor(test.df$fg_made)

# ==============================
# ----- 4) Baseline Forest ----- 
# ==============================
#The first step of finding our baseline forest is to perform bagging.
#The goal of bagging is to increase accuracy and lessen the risk of overfitting.
#Typically, we want B to be somewhere around 500-1000. For our data I choose 1000
#to have a larger # of trees.
#The second step of finding our baseline forest is to choose predictor variables and
#grow trees. Each of the B trees is fit using a randomly selected subset of predictor variables.
#This is shown to lessen correlation between the trees and increase accuracy. To choose how
#many predictor variables to choose we will call this m. m = sqrtK, K = # X Variables Available.
#m = sqrt(12) 3.464102, I rounded up to 4.

myforest<- randomForest(fg_made ~ .,
                        data = train.df, 
                        ntree = 1000, 
                        mtry = 4, 
                        importance = TRUE) 
myforest

# ===============================
# ----- 5) Define the model ----- 
# ===============================
#For defining the model we used classification and not regression because our
#Y variable fg_made is either 0 or 1.

rf_model <- rand_forest(mtry = tune(), 
                        trees = 1000) %>% 
  set_mode("classification") %>% 
  set_engine("randomForest")

# ==============================
# ----- 6) Create a recipe ----- 
# ==============================
#When creating our recipe using our training data set we used all X variables.

rf_rec <- recipe(fg_made ~ ., data = train.df)

# ==================================
# ----- 7) Create the workflow ----- 
# ==================================

rf_wf <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(rf_rec)

# ================================================
# ----- 8) Create folds for cross validation ----- 
# ================================================
#We choose to split our training data into 5 different folds.

set.seed(172)
folds <- vfold_cv(train.df, v = 5)

# =================================
# ----- 9) Tune random forest ----- 
# =================================
#We choose to limit the computational burden of tuning while examining a wide range
#by using the seq function. We started at the low end and ended at the high end
#with increasing by 3 each time.

?seq
rf_tuned <- tune_grid(
  rf_wf,
  resamples = folds, 
  grid = tibble(mtry = seq(1, 12, by =3)),
  metrics = metric_set( roc_auc),
  control = control_grid(save_pred = TRUE)
)

# ======================================================
# ----- 10) Extract AUC and/or OOB error estimates ----- 
# ======================================================
#After viewing the chart we can see a mtry of 4 maximizes the area under the curve.
#Our AUC = 0.730.

rf_results <- rf_tuned %>%
  collect_metrics()

ggplot(data = rf_results) +
  geom_line(aes(x = mtry, y = mean)) +
  geom_point(aes(x = mtry, y = mean)) +
  labs(x = "m (mtry) value", y = "Area Under the Curve (AUC)")+
  theme_bw() +
  scale_x_continuous(breaks = c(1:12))

# ============================
# ----- 11) Final Forest ----- 
# ============================
#This final forest is based on our tuning exercise with an mtry of 4.

best_params <- select_best(rf_tuned, metric = "roc_auc")

final_forest <- randomForest(fg_made ~.,
                             data = train.df,
                             ntree = 1000,
                             mtry = best_params %>% pull(mtry),
                             importance = TRUE)

# ================================
# ----- 12) Create ROC Curve ----- 
# ================================
#Our goal is prediction so we will create a ROC Curve.
#With our ROC Curve we will Predict probabilities of fg_made = 1 (positive class).

pi_hat <- predict(final_forest, test.df, type = "prob")[,"1"]

rocCurve <- roc(
  response = test.df$fg_made,
  predictor = pi_hat,
  levels = c("0", "1")
)
plot(rocCurve, print.thres = TRUE, print.auc = TRUE)

#If we set pi* equal to 0.839, we are estimated to get a specificity of 0.770.
#and sensitivity of 0.591.

#That is, we'll predict a missed FG 77% of the time when the FG is actually missed.
#Further, we'll predict a made FG 59.1% of the time when the FG is actually made.

# ========================================
# ----- 13) Variable Importance Plot ----- 
# ========================================
#Our goal is interpretation so we want to understand the relationships between
#our X variables and Y so we will create a variable importance plot.

varImpPlot(final_forest, type = 1)

#This gives us an ordered ranking of importance in terms of predictive ability
#of the variables. kick_distance is the most important with terms of predicting
#whether a FG is made.

# ================================================================
# ----- 14) Getting Directional Effects Using Random Forests ----- 
# ================================================================
#We are going to fit a logistic regression using the variables
#that are most important, as determined by the random forest.
#This series of logistic regressions will be created in order of variable
#importance plot and we will check the AIC to find best model.

fg_data$fg_made_bin <- ifelse(fg_data$fg_made == "1", 1, 0)

m1 <- glm(fg_made_bin ~ kick_distance,
          data = fg_data, family = binomial(link = "logit"))
AIC(m1) #19593.18

m2 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining,
          data = fg_data, family = binomial(link = "logit"))
AIC(m2) #19594.63
m3 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr,
          data = fg_data, family = binomial(link = "logit"))
AIC(m3) #19594.03
m4 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo,
          data = fg_data, family = binomial(link = "logit"))
AIC(m4) #19595.91
m5 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
            down,
          data = fg_data, family = binomial(link = "logit"))
AIC(m5) #19586.84
m6 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
            down + score_differential,
          data = fg_data, family = binomial(link = "logit"))
AIC(m6) #19588.72
m7 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
            down + score_differential + temp,
          data = fg_data, family = binomial(link = "logit"))
AIC(m7) #19545.62
m8 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
            down + score_differential + temp + wind,
          data = fg_data, family = binomial(link = "logit"))
AIC(m8) #19497.75
m9 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
            down + score_differential + temp + wind + grass,
          data = fg_data, family = binomial(link = "logit"))
AIC(m9) #19487.01
m10 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
            down + score_differential + temp + wind + grass + indoor,
          data = fg_data, family = binomial(link = "logit"))
AIC(m10) #19488.97
m11 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
             down + score_differential + temp + wind + grass + indoor + precipitation,
           data = fg_data, family = binomial(link = "logit"))
AIC(m11) #19474.46
m12 <- glm(fg_made_bin ~ kick_distance + game_seconds_remaining + qtr + ydstogo +
             down + score_differential + temp + wind + grass + indoor + precipitation
           + home,
           data = fg_data, family = binomial(link = "logit"))
AIC(m12) #19475.48

#We prefer model 11 because AIC got bigger after adding home.
#We will use model 11 (m4) for descriptive purposes.

# ========================================
# ----- 15) Interpretations Model 11 ----- 
# ========================================

summary(m11)
coef(m11)
coef(m11) %>% exp()
confint(m11)

#exp(-0.1012754 * 5) = exp(-0.506377) = 0.6026
#Holding all other variables constant, for every 5-yard increase in field-goal distance,
#the odds of making the kick decrease by a factor of 0.60.
#That is, odds drop by about 40%.

#exp(0.005475554 * 10) = exp(0.0547555) = 1.0563
#Holding all else constant, for every 10 degree F increase in temperature,
#the odds of making the kick increase by a factor of about 5.6%.

#exp(-0.01828812 * 5) = exp(-0.09144) = 0.9126
#Holding all else constant, for every 5 mph increase in wind speed,
#the odds of making the kick multiply by about 0.91 an 9% decrease in odds.

#exp(-0.1367309) = 0.8722
#On grass fields, the odds of making a field goal are about 0.87 times the odds
#on turf—roughly a 13% reduction—holding all other factors constant.

#exp(-0.2581087) = 0.7725
#In games with precipitation, the odds of making the kick are approximately
#0.77 times as large about a 23% reduction, holding all else constant.









