# ══════════════════════════════════════════════════════════════════════════════
# Lab_Week03.R — Resampling Methods: Cross-Validation & Bootstrap
# Machine Learning with R | Dr. Endri Raço
# ISLR Chapter 5
# ══════════════════════════════════════════════════════════════════════════════

library(ISLR2)
library(tidyverse)
library(boot)        # cv.glm() for CV, boot() for bootstrap


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: The Validation Set Approach
# ══════════════════════════════════════════════════════════════════════════════

# --- 1.1 Split Auto data randomly into two halves ---
# set.seed() ensures reproducibility: same split every time
set.seed(1)
train <- sample(392, 196)  # randomly select 196 indices out of 392

# --- 1.2 Fit linear, quadratic, cubic on training data ---
# The subset argument tells lm() to use only the training observations
lm_fit1 <- lm(mpg ~ horsepower, data = Auto, subset = train)
lm_fit2 <- lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train)
lm_fit3 <- lm(mpg ~ poly(horsepower, 3), data = Auto, subset = train)

# --- 1.3 Compute validation MSE on the held-out half ---
# predict() on the FULL dataset, then measure error only on -train (test)
mse1 <- mean((Auto$mpg - predict(lm_fit1, Auto))[-train]^2)
mse2 <- mean((Auto$mpg - predict(lm_fit2, Auto))[-train]^2)
mse3 <- mean((Auto$mpg - predict(lm_fit3, Auto))[-train]^2)

cat("Validation Set MSE:\n")
cat(sprintf("  Linear:    %.2f\n  Quadratic: %.2f\n  Cubic:     %.2f\n", mse1, mse2, mse3))
# Quadratic is much better than linear; cubic adds little

# --- 1.4 Repeat with a different seed to see variance ---
set.seed(2)
train2 <- sample(392, 196)
lm_fit2b <- lm(mpg ~ poly(horsepower, 2), data = Auto, subset = train2)
mse2b <- mean((Auto$mpg - predict(lm_fit2b, Auto))[-train2]^2)
cat(sprintf("  Quadratic with seed 2: %.2f  (vs %.2f with seed 1)\n", mse2b, mse2))
# Notice: different split gives different MSE — this is the variance problem


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Leave-One-Out Cross-Validation (LOOCV)
# ══════════════════════════════════════════════════════════════════════════════

# --- 2.1 LOOCV using cv.glm() ---
# We use glm() instead of lm() because cv.glm() requires a glm object.
# Without family argument, glm() fits OLS — identical to lm().
glm_fit <- glm(mpg ~ horsepower, data = Auto)
cv_err <- cv.glm(Auto, glm_fit)  # default K = n (LOOCV)
cat("\nLOOCV MSE (linear):", cv_err$delta[1], "\n")
# delta[1] = raw CV estimate, delta[2] = bias-corrected

# --- 2.2 LOOCV for polynomial degrees 1 through 10 ---
# This loop fits each polynomial degree and computes LOOCV error.
# The magic formula (5.2) makes this fast for linear models.
cv_error_loocv <- rep(0, 10)
for (d in 1:10) {
  glm_fit <- glm(mpg ~ poly(horsepower, d), data = Auto)
  cv_error_loocv[d] <- cv.glm(Auto, glm_fit)$delta[1]
}

cat("\nLOOCV MSE by polynomial degree:\n")
print(round(cv_error_loocv, 2))
# Sharp drop from d=1 (24.23) to d=2 (19.25), then flat

# --- 2.3 Visualize LOOCV curve ---
ggplot(data.frame(degree = 1:10, mse = cv_error_loocv), aes(x = degree, y = mse)) +
  geom_line(color = "#3B82F6", linewidth = 1) +
  geom_point(color = "#3B82F6", size = 3) +
  geom_point(data = data.frame(degree = 2, mse = cv_error_loocv[2]),
             color = "#10B981", size = 5, shape = 18) +
  labs(title = "LOOCV Error vs. Polynomial Degree (Auto data)",
       x = "Polynomial Degree", y = "LOOCV MSE") +
  theme_minimal()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: k-Fold Cross-Validation
# ══════════════════════════════════════════════════════════════════════════════

# --- 3.1 10-fold CV for polynomial degrees 1 to 10 ---
set.seed(17)
cv_error_10fold <- rep(0, 10)
for (d in 1:10) {
  glm_fit <- glm(mpg ~ poly(horsepower, d), data = Auto)
  # K = 10 specifies 10-fold CV (much faster than LOOCV for complex models)
  cv_error_10fold[d] <- cv.glm(Auto, glm_fit, K = 10)$delta[1]
}

cat("\n10-fold CV MSE by polynomial degree:\n")
print(round(cv_error_10fold, 2))

# --- 3.2 Compare LOOCV vs 10-fold CV ---
comparison <- data.frame(
  degree = rep(1:10, 2),
  mse    = c(cv_error_loocv, cv_error_10fold),
  method = rep(c("LOOCV", "10-fold CV"), each = 10)
)

ggplot(comparison, aes(x = degree, y = mse, color = method)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = c("LOOCV" = "#3B82F6", "10-fold CV" = "#F59E0B")) +
  labs(title = "LOOCV vs. 10-Fold CV on Auto Data",
       x = "Polynomial Degree", y = "CV MSE", color = "Method") +
  theme_minimal()
# Both agree: quadratic (d=2) is the sweet spot

# --- 3.3 Variability of 10-fold CV across different splits ---
# Run 10-fold CV 9 times with different random seeds to show stability
cv_matrix <- matrix(0, nrow = 9, ncol = 10)
for (s in 1:9) {
  set.seed(s)
  for (d in 1:10) {
    glm_fit <- glm(mpg ~ poly(horsepower, d), data = Auto)
    cv_matrix[s, d] <- cv.glm(Auto, glm_fit, K = 10)$delta[1]
  }
}

# Plot all 9 curves — they are very similar (much less variance than validation set)
cv_long <- as.data.frame(cv_matrix)
names(cv_long) <- paste0("d", 1:10)
cv_long$seed <- 1:9
cv_long <- pivot_longer(cv_long, cols = -seed, names_to = "degree", values_to = "mse")
cv_long$degree <- as.numeric(gsub("d", "", cv_long$degree))

ggplot(cv_long, aes(x = degree, y = mse, group = seed)) +
  geom_line(alpha = 0.4, color = "#3B82F6") +
  labs(title = "9 Runs of 10-Fold CV: Much Less Variance Than Validation Set",
       x = "Polynomial Degree", y = "10-fold CV MSE") +
  theme_minimal()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: The Bootstrap
# ══════════════════════════════════════════════════════════════════════════════

# --- 4.1 Portfolio allocation: estimate alpha and its SE ---
# The Portfolio dataset has two assets X and Y.
# Optimal allocation alpha = (Var(Y) - Cov(X,Y)) / (Var(X) + Var(Y) - 2*Cov(X,Y))
data(Portfolio)

# Define a function that computes alpha from a subset of rows.
# boot() will call this function with different resampled index vectors.
alpha.fn <- function(data, index) {
  X <- data$X[index]
  Y <- data$Y[index]
  (var(Y) - cov(X, Y)) / (var(X) + var(Y) - 2 * cov(X, Y))
}

# Original estimate
cat("\nOriginal alpha:", alpha.fn(Portfolio, 1:100), "\n")

# --- 4.2 Run bootstrap with B = 1000 replicates ---
set.seed(1)
boot_alpha <- boot(Portfolio, alpha.fn, R = 1000)
print(boot_alpha)
# t1 = original estimate (~0.576)
# SE = standard error (~0.089)

# --- 4.3 Visualize the bootstrap distribution ---
ggplot(data.frame(alpha = boot_alpha$t), aes(x = alpha)) +
  geom_histogram(bins = 40, fill = "#3B82F6", color = "white", alpha = 0.8) +
  geom_vline(xintercept = boot_alpha$t0, color = "#EF4444", linewidth = 1, linetype = "dashed") +
  labs(title = "Bootstrap Distribution of Alpha (B = 1000)",
       subtitle = sprintf("alpha = %.3f, SE = %.3f", boot_alpha$t0, sd(boot_alpha$t)),
       x = "Bootstrap alpha*", y = "Count") +
  theme_minimal()

# --- 4.4 Bootstrap confidence interval ---
boot.ci(boot_alpha, type = c("norm", "perc"))
# norm = normal approximation, perc = percentile method


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Bootstrap for Regression Coefficients
# ══════════════════════════════════════════════════════════════════════════════

# --- 5.1 Define function that returns regression coefficients ---
boot.fn <- function(data, index) {
  coef(lm(mpg ~ horsepower, data = data, subset = index))
}

# Original coefficients
boot.fn(Auto, 1:392)

# --- 5.2 Run bootstrap ---
set.seed(1)
boot_reg <- boot(Auto, boot.fn, R = 1000)
print(boot_reg)

# --- 5.3 Compare formula SEs vs bootstrap SEs ---
# Formula-based SEs (assume linear model is correct)
formula_se <- summary(lm(mpg ~ horsepower, data = Auto))$coefficients[, "Std. Error"]

# Bootstrap SEs (model-free)
bootstrap_se <- apply(boot_reg$t, 2, sd)

cat("\n=== SE Comparison: mpg ~ horsepower ===\n")
cat(sprintf("  Intercept: Formula = %.4f, Bootstrap = %.4f\n", formula_se[1], bootstrap_se[1]))
cat(sprintf("  Slope:     Formula = %.4f, Bootstrap = %.4f\n", formula_se[2], bootstrap_se[2]))
# Bootstrap SEs are LARGER because the true relationship is non-linear,
# violating the assumptions of the formula-based SEs.

# --- 5.4 Repeat with quadratic model ---
boot.fn2 <- function(data, index) {
  coef(lm(mpg ~ horsepower + I(horsepower^2), data = data, subset = index))
}

boot_reg2 <- boot(Auto, boot.fn2, R = 1000)
formula_se2 <- summary(lm(mpg ~ horsepower + I(horsepower^2), data = Auto))$coefficients[, "Std. Error"]
bootstrap_se2 <- apply(boot_reg2$t, 2, sd)

cat("\n=== SE Comparison: mpg ~ horsepower + horsepower^2 ===\n")
cat(sprintf("  Intercept: Formula = %.4f, Bootstrap = %.4f\n", formula_se2[1], bootstrap_se2[1]))
cat(sprintf("  hp:        Formula = %.6f, Bootstrap = %.6f\n", formula_se2[2], bootstrap_se2[2]))
cat(sprintf("  hp^2:      Formula = %.6f, Bootstrap = %.6f\n", formula_se2[3], bootstrap_se2[3]))
# With the quadratic model, formula and bootstrap SEs are much closer!
# This confirms the model is a better fit for the data.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EXERCISES
# ══════════════════════════════════════════════════════════════════════════════

# ── Exercise 1: Validation Set on Default (ISLR 5.4 Q5) ─────────────────
## YOUR CODE HERE ##

# ── Exercise 2: Bootstrap SEs for Logistic (ISLR 5.4 Q6) ────────────────
## YOUR CODE HERE ##

# ── Exercise 3: LOOCV Polynomial Selection (ISLR 5.4 Q5-6) ──────────────
## YOUR CODE HERE ##

# ── Exercise 4: Bootstrap Portfolio (ISLR 5.4 Q8) ────────────────────────
## YOUR CODE HERE ##


# ══════════════════════════════════════════════════════════════════════════════
# SOLUTION KEYS
# ══════════════════════════════════════════════════════════════════════════════

# ── Solution: Exercise 1 (Validation Set on Default) ─────────────────────

# (a) Fit logistic on full data
glm_default <- glm(default ~ income + balance, data = Default, family = binomial)
summary(glm_default)

# (b) Validation set approach
set.seed(1)
n <- nrow(Default)
train_idx <- sample(n, n / 2)

glm_train <- glm(default ~ income + balance, data = Default, subset = train_idx, family = binomial)
probs     <- predict(glm_train, Default[-train_idx, ], type = "response")
preds     <- ifelse(probs > 0.5, "Yes", "No")
val_error <- mean(preds != Default$default[-train_idx])
cat("\nExercise 1: Validation error (seed 1):", val_error, "\n")

# (c) Repeat with different seeds
for (s in c(42, 123, 456)) {
  set.seed(s)
  tr <- sample(n, n / 2)
  fit <- glm(default ~ income + balance, data = Default, subset = tr, family = binomial)
  pr  <- ifelse(predict(fit, Default[-tr, ], type = "response") > 0.5, "Yes", "No")
  cat(sprintf("  Seed %3d: error = %.4f\n", s, mean(pr != Default$default[-tr])))
}
# Errors vary by ~0.3-0.5% across splits

# (d) With student
glm_student <- glm(default ~ income + balance + student, data = Default,
                     subset = train_idx, family = binomial)
probs_s <- predict(glm_student, Default[-train_idx, ], type = "response")
preds_s <- ifelse(probs_s > 0.5, "Yes", "No")
cat("  With student:", mean(preds_s != Default$default[-train_idx]), "\n")
# Minimal difference — student doesn't help


# ── Solution: Exercise 2 (Bootstrap SEs for Logistic) ────────────────────

boot.logistic <- function(data, index) {
  fit <- glm(default ~ income + balance, data = data, subset = index, family = binomial)
  coef(fit)
}

set.seed(1)
boot_logistic <- boot(Default, boot.logistic, R = 1000)

# Formula SEs
formula_se_log <- summary(glm_default)$coefficients[, "Std. Error"]
bootstrap_se_log <- apply(boot_logistic$t, 2, sd)

cat("\nExercise 2: Logistic Regression SE Comparison\n")
cat(sprintf("  Intercept: Formula = %.6f, Bootstrap = %.6f\n",
            formula_se_log[1], bootstrap_se_log[1]))
cat(sprintf("  Income:    Formula = %.8f, Bootstrap = %.8f\n",
            formula_se_log[2], bootstrap_se_log[2]))
cat(sprintf("  Balance:   Formula = %.8f, Bootstrap = %.8f\n",
            formula_se_log[3], bootstrap_se_log[3]))
# Very similar — logistic model assumptions are reasonable for this data


# ── Solution: Exercise 4 (Bootstrap Portfolio) ───────────────────────────

set.seed(1)
boot_port <- boot(Portfolio, alpha.fn, R = 1000)
cat("\nExercise 4: Portfolio Bootstrap\n")
cat(sprintf("  alpha = %.3f, SE = %.3f\n", boot_port$t0, sd(boot_port$t)))
cat("  95%% CI (percentile):\n")
print(boot.ci(boot_port, type = "perc"))


# ══════════════════════════════════════════════════════════════════════════════
# END OF LAB_WEEK03.R
# ══════════════════════════════════════════════════════════════════════════════
