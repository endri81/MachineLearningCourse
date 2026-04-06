# ══════════════════════════════════════════════════════════════════════════════
# Lab_Week04.R — Linear Model Selection & Regularization
# Machine Learning with R | Dr. Endri Raço | ISLR Chapter 6
# ══════════════════════════════════════════════════════════════════════════════

library(ISLR2)
library(tidyverse)
library(glmnet)    # Ridge, Lasso, Elastic Net
library(leaps)     # Best subset / stepwise selection


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Data Preparation — Hitters Dataset
# ══════════════════════════════════════════════════════════════════════════════

Hitters <- na.omit(ISLR2::Hitters)  # Remove 59 NAs → n = 263
cat("Hitters: n =", nrow(Hitters), ", p =", ncol(Hitters) - 1, "\n")

# glmnet requires a numeric matrix (no factors, no response column)
x <- model.matrix(Salary ~ ., data = Hitters)[, -1]  # drop intercept column
y <- Hitters$Salary

# Train/test split (for final evaluation)
set.seed(1)
train <- sample(nrow(x), nrow(x) * 0.7)
x_train <- x[train, ];  y_train <- y[train]
x_test  <- x[-train, ]; y_test  <- y[-train]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Best Subset Selection
# ══════════════════════════════════════════════════════════════════════════════

# regsubsets() fits all possible models for each number of predictors
reg_full <- regsubsets(Salary ~ ., data = Hitters, nvmax = 19)
reg_summary <- summary(reg_full)

# Plot BIC, Cp, and Adjusted R² to choose # predictors
par(mfrow = c(1, 3))
plot(reg_summary$bic, type = "b", xlab = "# Predictors", ylab = "BIC", pch = 19, col = "#3B82F6")
points(which.min(reg_summary$bic), min(reg_summary$bic), col = "red", pch = 19, cex = 2)

plot(reg_summary$cp, type = "b", xlab = "# Predictors", ylab = "Cp", pch = 19, col = "#3B82F6")
points(which.min(reg_summary$cp), min(reg_summary$cp), col = "red", pch = 19, cex = 2)

plot(reg_summary$adjr2, type = "b", xlab = "# Predictors", ylab = "Adj R²", pch = 19, col = "#3B82F6")
points(which.max(reg_summary$adjr2), max(reg_summary$adjr2), col = "red", pch = 19, cex = 2)
par(mfrow = c(1, 1))

cat("BIC selects", which.min(reg_summary$bic), "predictors\n")
cat("Cp selects", which.min(reg_summary$cp), "predictors\n")
coef(reg_full, which.min(reg_summary$bic))  # BIC-selected coefficients


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Ridge Regression
# ══════════════════════════════════════════════════════════════════════════════

# --- 3.1 Fit ridge over a grid of lambda values ---
# alpha = 0 specifies ridge; glmnet automatically standardizes x
ridge_mod <- glmnet(x_train, y_train, alpha = 0)

# Coefficient path plot: how coefficients shrink as lambda increases
plot(ridge_mod, xvar = "lambda", label = TRUE)
title("Ridge Regression: Coefficient Path")

# --- 3.2 Cross-validation to choose lambda ---
set.seed(1)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10)
plot(cv_ridge)
title("Ridge: 10-Fold CV")

cat("Ridge lambda_min:", cv_ridge$lambda.min, "\n")
cat("Ridge lambda_1se:", cv_ridge$lambda.1se, "\n")

# --- 3.3 Evaluate on test set ---
ridge_pred <- predict(cv_ridge, s = cv_ridge$lambda.min, newx = x_test)
ridge_mse <- mean((y_test - ridge_pred)^2)
cat("Ridge test MSE:", round(ridge_mse), "\n")

# Ridge coefficients (all 19 are non-zero)
coef(cv_ridge, s = "lambda.min")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: The Lasso
# ══════════════════════════════════════════════════════════════════════════════

# --- 4.1 Fit lasso (alpha = 1) ---
lasso_mod <- glmnet(x_train, y_train, alpha = 1)
plot(lasso_mod, xvar = "lambda", label = TRUE)
title("Lasso: Coefficient Path")

# --- 4.2 CV for lambda ---
set.seed(1)
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1, nfolds = 10)
plot(cv_lasso)
title("Lasso: 10-Fold CV")

cat("\nLasso lambda_min:", cv_lasso$lambda.min, "\n")
cat("Lasso lambda_1se:", cv_lasso$lambda.1se, "\n")

# --- 4.3 Coefficients at lambda.min and lambda.1se ---
lasso_coef_min <- coef(cv_lasso, s = "lambda.min")
lasso_coef_1se <- coef(cv_lasso, s = "lambda.1se")

cat("Non-zero at lambda_min:", sum(lasso_coef_min != 0) - 1, "\n")  # -1 for intercept
cat("Non-zero at lambda_1se:", sum(lasso_coef_1se != 0) - 1, "\n")

# Show non-zero coefficients
print(lasso_coef_1se[lasso_coef_1se[, 1] != 0, ])

# --- 4.4 Test MSE ---
lasso_pred <- predict(cv_lasso, s = cv_lasso$lambda.min, newx = x_test)
lasso_mse <- mean((y_test - lasso_pred)^2)
cat("Lasso test MSE:", round(lasso_mse), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Elastic Net
# ══════════════════════════════════════════════════════════════════════════════

# --- 5.1 Find best alpha via CV ---
alpha_grid <- seq(0, 1, by = 0.1)
cv_results <- data.frame(alpha = alpha_grid, min_cv = NA)

for (i in seq_along(alpha_grid)) {
  set.seed(1)
  cv_fit <- cv.glmnet(x_train, y_train, alpha = alpha_grid[i])
  cv_results$min_cv[i] <- min(cv_fit$cvm)
}

cat("\nElastic Net: CV MSE by alpha:\n")
print(cv_results)

best_alpha <- cv_results$alpha[which.min(cv_results$min_cv)]
cat("Best alpha:", best_alpha, "\n")

# --- 5.2 Fit elastic net with best alpha ---
set.seed(1)
cv_enet <- cv.glmnet(x_train, y_train, alpha = best_alpha)
enet_pred <- predict(cv_enet, s = cv_enet$lambda.min, newx = x_test)
enet_mse <- mean((y_test - enet_pred)^2)
cat("Elastic Net test MSE:", round(enet_mse), "\n")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: Comparison
# ══════════════════════════════════════════════════════════════════════════════

ols_mse <- mean((y_test - predict(lm(Salary ~ ., data = Hitters[train, ]), Hitters[-train, ]))^2)

cat("\n=== Method Comparison (Hitters) ===\n")
cat(sprintf("  OLS:          %6.0f  (19 predictors)\n", ols_mse))
cat(sprintf("  Ridge:        %6.0f  (19 predictors)\n", ridge_mse))
cat(sprintf("  Lasso:        %6.0f  (%d predictors)\n", lasso_mse, sum(lasso_coef_min != 0) - 1))
cat(sprintf("  Elastic Net:  %6.0f  (alpha = %.1f)\n", enet_mse, best_alpha))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: EXERCISES & SOLUTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Exercise 1: College dataset (ISLR 6.6 Q11) ──────────────────────────
# Predict Apps using all other predictors. Compare OLS, Ridge, Lasso, PCR.

data(College)
set.seed(42)
train_c <- sample(nrow(College), nrow(College) * 0.7)

x_col <- model.matrix(Apps ~ ., data = College)[, -1]
y_col <- College$Apps

# OLS
ols_col <- lm(Apps ~ ., data = College[train_c, ])
ols_pred_col <- predict(ols_col, College[-train_c, ])
cat("\nCollege OLS test MSE:", round(mean((y_col[-train_c] - ols_pred_col)^2)), "\n")

# Ridge
set.seed(1)
cv_ridge_col <- cv.glmnet(x_col[train_c, ], y_col[train_c], alpha = 0)
ridge_pred_col <- predict(cv_ridge_col, s = "lambda.min", newx = x_col[-train_c, ])
cat("College Ridge test MSE:", round(mean((y_col[-train_c] - ridge_pred_col)^2)), "\n")

# Lasso
set.seed(1)
cv_lasso_col <- cv.glmnet(x_col[train_c, ], y_col[train_c], alpha = 1)
lasso_pred_col <- predict(cv_lasso_col, s = "lambda.min", newx = x_col[-train_c, ])
cat("College Lasso test MSE:", round(mean((y_col[-train_c] - lasso_pred_col)^2)), "\n")
cat("College Lasso predictors:", sum(coef(cv_lasso_col, s = "lambda.min") != 0) - 1, "\n")

# Show which predictors Lasso keeps
lasso_coef_col <- coef(cv_lasso_col, s = "lambda.min")
cat("Selected predictors:\n")
print(lasso_coef_col[lasso_coef_col[, 1] != 0, ])


# ══════════════════════════════════════════════════════════════════════════════
# END OF LAB_WEEK04.R
# ══════════════════════════════════════════════════════════════════════════════
