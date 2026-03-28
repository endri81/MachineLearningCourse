# ══════════════════════════════════════════════════════════════════════════════
# Lab_Week02.R — Classification: Logistic Reg, LDA, QDA, NB, KNN
# Machine Learning with R | Dr. Endri Raço
# ISLR Chapter 4
# ══════════════════════════════════════════════════════════════════════════════

# ── SETUP ─────────────────────────────────────────────────────────────────────
library(ISLR2)
library(tidyverse)
library(MASS)       # lda(), qda()
library(e1071)      # naiveBayes()
library(class)      # knn()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Exploratory Data Analysis — Default Dataset
# ══════════════════════════════════════════════════════════════════════════════

data(Default)
str(Default)        # 10,000 obs: default, student, balance, income
summary(Default)

# Default rate: only 3.3% default — heavily imbalanced
table(Default$default)
prop.table(table(Default$default))

# Boxplots reveal that balance is the strongest discriminator:
# defaulters carry much higher balances on average
ggplot(Default, aes(x = default, y = balance, fill = default)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("#3B82F6", "#F97316")) +
  labs(title = "Balance by Default Status", x = "Default", y = "Balance") +
  theme_minimal() + theme(legend.position = "none")

# Income shows less separation between classes
ggplot(Default, aes(x = default, y = income, fill = default)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("#3B82F6", "#F97316")) +
  labs(title = "Income by Default Status") +
  theme_minimal() + theme(legend.position = "none")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

# --- 2.1 Simple logistic regression: default ~ balance ---
# glm() with family = binomial fits logistic regression via MLE.
# The response must be a factor or 0/1 numeric.
glm_balance <- glm(default ~ balance, data = Default, family = binomial)
summary(glm_balance)

# beta_1 = 0.0055: each $1 increase in balance increases log-odds by 0.0055
# Odds ratio: exp(0.0055) = 1.0055 per dollar, or exp(0.55) = 1.73 per $100

# --- 2.2 Predict probability at specific balance values ---
# type = "response" returns p(X), not the log-odds
new_balances <- data.frame(balance = c(500, 1000, 1500, 2000, 2500))
new_balances$prob <- predict(glm_balance, new_balances, type = "response")
new_balances
# Notice the non-linear jump: small prob at $500, ~59% at $2000

# --- 2.3 Multiple logistic regression ---
glm_full <- glm(default ~ balance + income + student, data = Default, family = binomial)
summary(glm_full)
# Key insight: student coefficient is NEGATIVE (-0.65) in multiple regression
# but POSITIVE in simple regression — confounding effect
# Students carry higher balances; controlling for balance, they default less

# --- 2.4 Confusion matrix at threshold 0.5 ---
# predict on training data to see the confusion matrix
probs_train <- predict(glm_full, type = "response")
pred_train  <- ifelse(probs_train > 0.5, "Yes", "No")

# table() creates the confusion matrix: rows = actual, cols = predicted
table(Actual = Default$default, Predicted = pred_train)

# Overall accuracy
mean(pred_train == Default$default)

# Sensitivity (true positive rate): how many defaulters did we catch?
tp <- sum(pred_train == "Yes" & Default$default == "Yes")
fn <- sum(pred_train == "No"  & Default$default == "Yes")
cat("Sensitivity:", round(tp / (tp + fn), 3), "\n")
# Sensitivity is low (~30%) because most predictions are "No"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Smarket Data — Train/Test Split
# ══════════════════════════════════════════════════════════════════════════════

data(Smarket)
str(Smarket)   # 1250 obs: Year, Lag1-Lag5, Volume, Today, Direction

# Split: train on 2001-2004, test on 2005
train_idx   <- Smarket$Year < 2005
Smarket_train <- Smarket[train_idx, ]
Smarket_test  <- Smarket[!train_idx, ]
Direction_test <- Smarket_test$Direction

# --- 3.1 Logistic regression (Lag1 + Lag2 only) ---
glm_smarket <- glm(Direction ~ Lag1 + Lag2, data = Smarket_train, family = binomial)
probs_test  <- predict(glm_smarket, Smarket_test, type = "response")
pred_test   <- ifelse(probs_test > 0.5, "Up", "Down")

table(Actual = Direction_test, Predicted = pred_test)
cat("Logistic test accuracy:", mean(pred_test == Direction_test), "\n")

# --- 3.2 LDA ---
lda_fit  <- lda(Direction ~ Lag1 + Lag2, data = Smarket_train)
lda_pred <- predict(lda_fit, Smarket_test)

table(Actual = Direction_test, Predicted = lda_pred$class)
cat("LDA test accuracy:", mean(lda_pred$class == Direction_test), "\n")

# --- 3.3 QDA ---
qda_fit  <- qda(Direction ~ Lag1 + Lag2, data = Smarket_train)
qda_pred <- predict(qda_fit, Smarket_test)

table(Actual = Direction_test, Predicted = qda_pred$class)
cat("QDA test accuracy:", mean(qda_pred$class == Direction_test), "\n")

# --- 3.4 Naive Bayes ---
nb_fit  <- naiveBayes(Direction ~ Lag1 + Lag2, data = Smarket_train)
nb_pred <- predict(nb_fit, Smarket_test)

table(Actual = Direction_test, Predicted = nb_pred)
cat("Naive Bayes accuracy:", mean(nb_pred == Direction_test), "\n")

# --- 3.5 KNN (K = 1, 3, 5, 10) ---
# knn() requires matrix inputs, not data frames with factors
train_X <- as.matrix(Smarket_train[, c("Lag1", "Lag2")])
test_X  <- as.matrix(Smarket_test[, c("Lag1", "Lag2")])
train_Y <- Smarket_train$Direction

for (k in c(1, 3, 5, 10)) {
  knn_pred <- knn(train_X, test_X, train_Y, k = k)
  acc <- mean(knn_pred == Direction_test)
  cat(sprintf("KNN K=%2d accuracy: %.3f\n", k, acc))
}

# --- 3.6 Compare all methods ---
cat("\n=== Smarket: Method Comparison ===\n")
cat("Logistic:", mean(pred_test == Direction_test), "\n")
cat("LDA:     ", mean(lda_pred$class == Direction_test), "\n")
cat("QDA:     ", mean(qda_pred$class == Direction_test), "\n")
cat("NB:      ", mean(nb_pred == Direction_test), "\n")
# QDA typically wins on Smarket data (~60%)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: ROC Curve (Manual Construction)
# ══════════════════════════════════════════════════════════════════════════════

# Using the Default data logistic model
probs <- predict(glm_balance, type = "response")
actual <- Default$default == "Yes"

# Compute TPR and FPR at many thresholds
thresholds <- seq(0, 1, by = 0.01)
roc_data <- data.frame(threshold = thresholds, TPR = NA, FPR = NA)

for (i in seq_along(thresholds)) {
  pred_pos <- probs >= thresholds[i]
  tp <- sum(pred_pos & actual)
  fp <- sum(pred_pos & !actual)
  fn <- sum(!pred_pos & actual)
  tn <- sum(!pred_pos & !actual)
  roc_data$TPR[i] <- tp / (tp + fn)
  roc_data$FPR[i] <- fp / (fp + tn)
}

ggplot(roc_data, aes(x = FPR, y = TPR)) +
  geom_line(color = "#3B82F6", linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey50") +
  labs(title = "ROC Curve: Default ~ Balance",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()

# Approximate AUC using trapezoidal rule
roc_sorted <- roc_data[order(roc_data$FPR), ]
auc <- sum(diff(roc_sorted$FPR) * (head(roc_sorted$TPR, -1) + tail(roc_sorted$TPR, -1)) / 2)
cat("Approximate AUC:", round(auc, 3), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EXERCISES
# ══════════════════════════════════════════════════════════════════════════════

# ── Exercise 1: Default (ISLR 4.8 Q13) ──────────────────────────────────
# (a) Fit logistic: default ~ income + balance
# (b) Split train/test, compute test error
# (c) Repeat 3 times with different seeds
# (d) Add student — does it improve?

## YOUR CODE HERE ##


# ── Exercise 2: Weekly (ISLR 4.8 Q11) ───────────────────────────────────
# (a) EDA
# (b) Logistic with all Lag + Volume
# (c) Confusion matrix
# (d) Logistic with Lag2 only (train: 1990-2008, test: 2009-2010)
# (e) Repeat with LDA, QDA, KNN K=5

## YOUR CODE HERE ##


# ── Exercise 3: Boston crime (ISLR 4.8 Q16) ─────────────────────────────
# Create binary: crim > median
# Fit logistic, LDA, KNN with various K

## YOUR CODE HERE ##


# ══════════════════════════════════════════════════════════════════════════════
# SOLUTION KEYS
# ══════════════════════════════════════════════════════════════════════════════

# ── Solution: Exercise 1 ─────────────────────────────────────────────────

# (a)
set.seed(1)
train <- sample(nrow(Default), nrow(Default) * 0.7)
ex1_fit <- glm(default ~ income + balance, data = Default, subset = train, family = binomial)
summary(ex1_fit)

# (b)
ex1_probs <- predict(ex1_fit, Default[-train, ], type = "response")
ex1_pred  <- ifelse(ex1_probs > 0.5, "Yes", "No")
cat("Test error rate:", mean(ex1_pred != Default$default[-train]), "\n")

# (c) Repeat with different seeds
for (s in c(2, 42, 123)) {
  set.seed(s)
  tr <- sample(nrow(Default), nrow(Default) * 0.7)
  fit <- glm(default ~ income + balance, data = Default, subset = tr, family = binomial)
  pr  <- ifelse(predict(fit, Default[-tr, ], type = "response") > 0.5, "Yes", "No")
  cat(sprintf("  Seed %3d: test error = %.4f\n", s, mean(pr != Default$default[-tr])))
}

# (d) With student
ex1_fit2 <- glm(default ~ income + balance + student, data = Default, subset = train, family = binomial)
ex1_pred2 <- ifelse(predict(ex1_fit2, Default[-train, ], type = "response") > 0.5, "Yes", "No")
cat("With student, test error:", mean(ex1_pred2 != Default$default[-train]), "\n")
# Minimal improvement — student adds little beyond balance


# ── Solution: Exercise 2 ─────────────────────────────────────────────────

data(Weekly)
# (a)
summary(Weekly)
cor(Weekly[, -9])  # Direction is factor, exclude it

# (b)
glm_weekly <- glm(Direction ~ Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Volume,
                   data = Weekly, family = binomial)
summary(glm_weekly)  # Only Lag2 is marginally significant

# (c)
weekly_pred <- ifelse(predict(glm_weekly, type = "response") > 0.5, "Up", "Down")
table(Actual = Weekly$Direction, Predicted = weekly_pred)
cat("Training accuracy:", mean(weekly_pred == Weekly$Direction), "\n")

# (d) Train 1990-2008, test 2009-2010
train_w <- Weekly$Year <= 2008
test_w  <- Weekly[!train_w, ]

glm_w2 <- glm(Direction ~ Lag2, data = Weekly, subset = train_w, family = binomial)
pred_w2 <- ifelse(predict(glm_w2, test_w, type = "response") > 0.5, "Up", "Down")
table(Actual = test_w$Direction, Predicted = pred_w2)
cat("Logistic (Lag2) accuracy:", mean(pred_w2 == test_w$Direction), "\n")

# (e) LDA
lda_w <- lda(Direction ~ Lag2, data = Weekly, subset = train_w)
lda_pw <- predict(lda_w, test_w)$class
cat("LDA accuracy:", mean(lda_pw == test_w$Direction), "\n")

# QDA
qda_w <- qda(Direction ~ Lag2, data = Weekly, subset = train_w)
qda_pw <- predict(qda_w, test_w)$class
cat("QDA accuracy:", mean(qda_pw == test_w$Direction), "\n")

# KNN K=5
train_lag2 <- as.matrix(Weekly$Lag2[train_w])
test_lag2  <- as.matrix(test_w$Lag2)
knn_pw <- knn(train_lag2, test_lag2, Weekly$Direction[train_w], k = 5)
cat("KNN K=5 accuracy:", mean(knn_pw == test_w$Direction), "\n")


# ── Solution: Exercise 3 ─────────────────────────────────────────────────

data(Boston)
# Create binary response
Boston$high_crime <- ifelse(Boston$crim > median(Boston$crim), 1, 0)

set.seed(42)
train_b <- sample(nrow(Boston), nrow(Boston) * 0.7)
test_b  <- Boston[-train_b, ]

# Logistic
glm_b <- glm(high_crime ~ nox + rad + tax + lstat + dis + age,
              data = Boston, subset = train_b, family = binomial)
pred_b <- ifelse(predict(glm_b, test_b, type = "response") > 0.5, 1, 0)
cat("Boston Logistic accuracy:", mean(pred_b == test_b$high_crime), "\n")

# LDA
lda_b <- lda(high_crime ~ nox + rad + tax + lstat + dis + age,
              data = Boston, subset = train_b)
lda_pb <- predict(lda_b, test_b)$class
cat("Boston LDA accuracy:", mean(lda_pb == test_b$high_crime), "\n")

# KNN K=3
train_X_b <- scale(Boston[train_b, c("nox", "rad", "tax", "lstat", "dis", "age")])
test_X_b  <- scale(test_b[, c("nox", "rad", "tax", "lstat", "dis", "age")],
                    center = attr(train_X_b, "scaled:center"),
                    scale  = attr(train_X_b, "scaled:scale"))
knn_pb <- knn(train_X_b, test_X_b, Boston$high_crime[train_b], k = 3)
cat("Boston KNN K=3 accuracy:", mean(knn_pb == test_b$high_crime), "\n")
# KNN often performs best here due to non-linear boundaries

# ══════════════════════════════════════════════════════════════════════════════
# END OF LAB_WEEK02.R
# ══════════════════════════════════════════════════════════════════════════════
