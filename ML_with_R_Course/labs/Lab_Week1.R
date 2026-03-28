# ══════════════════════════════════════════════════════════════════════════════
# Lab_Week1.R — Introduction to Statistical Learning & Linear Regression
# Machine Learning with R | Dr. Endri Raço
# ISLR Chapters 2–3
# Reference: github.com/endri81/ISLRv2-solutions
# ══════════════════════════════════════════════════════════════════════════════

# ── SETUP: Install and load required packages ─────────────────────────────
# install.packages(c("ISLR2", "tidyverse", "car", "GGally"))

library(ISLR2)    # Contains all textbook datasets (Auto, Boston, Carseats, etc.)
library(tidyverse) # For ggplot2, dplyr, and tidy data manipulation
library(car)       # For vif() — Variance Inflation Factor computation


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Exploratory Data Analysis — Auto Dataset (Lines 1–35)
# ══════════════════════════════════════════════════════════════════════════════

# --- 1.1 Load and inspect the Auto dataset ---
# Auto contains fuel efficiency data for 392 vehicles from the 1970s–80s.
# We want to understand what drives mpg (miles per gallon).
data(Auto)

# Examine the structure: 392 observations, 9 variables
str(Auto)

# Quick statistical summary to see ranges, quartiles, and potential issues
summary(Auto)

# First few rows — notice 'name' is a character (car model), not a predictor
head(Auto)

# --- 1.2 Visualize the response variable (mpg) ---
# Histogram reveals the distribution shape; mpg is right-skewed
ggplot(Auto, aes(x = mpg)) +
  geom_histogram(bins = 25, fill = "#3B82F6", color = "white", alpha = 0.8) +
  labs(title = "Distribution of Miles Per Gallon",
       x = "MPG", y = "Count") +
  theme_minimal()

# --- 1.3 Scatterplot: mpg vs horsepower ---
# This is the key relationship we will model with SLR.
# Notice the non-linear (curved) pattern — more power means worse fuel economy,
# but the relationship flattens at high horsepower.
ggplot(Auto, aes(x = horsepower, y = mpg)) +
  geom_point(alpha = 0.5, color = "#1E2761") +
  labs(title = "MPG vs. Horsepower",
       x = "Horsepower", y = "Miles Per Gallon") +
  theme_minimal()

# --- 1.4 Correlation matrix (numeric variables only) ---
# Correlation quantifies linear association between every pair of variables.
# Strong correlations with mpg: weight (-0.83), displacement (-0.80), horsepower (-0.78)
Auto_numeric <- Auto %>% select(-name)
round(cor(Auto_numeric), 2)

# --- 1.5 Pairwise scatterplot matrix ---
# Visualize all bivariate relationships at once. Look for: linear trends,
# clusters, non-linear patterns, and potential collinearity between predictors.
pairs(Auto_numeric, pch = 16, cex = 0.4, col = "#1E276180")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Simple Linear Regression — mpg ~ horsepower (Lines 40–80)
# ══════════════════════════════════════════════════════════════════════════════

# --- 2.1 Fit a simple linear regression ---
# lm() fits Y = beta_0 + beta_1 * X + epsilon using ordinary least squares.
# The formula mpg ~ horsepower tells R that mpg is the response and horsepower
# is the single predictor.
slr_fit <- lm(mpg ~ horsepower, data = Auto)

# --- 2.2 Examine the summary output ---
# Coefficients table: Estimate, Std. Error, t value, Pr(>|t|)
# beta_0 (Intercept) ≈ 39.94: predicted mpg when horsepower = 0 (extrapolation!)
# beta_1 (horsepower) ≈ -0.158: each unit increase in hp → mpg drops by 0.158
# p-value < 2e-16: extremely strong evidence of a relationship
# R-squared ≈ 0.606: horsepower explains about 61% of mpg variability
# RSE ≈ 4.91: typical prediction error is about 4.9 mpg
summary(slr_fit)

# --- 2.3 Extract specific model quantities ---
coef(slr_fit)              # Coefficients: intercept and slope
confint(slr_fit)           # 95% confidence intervals for beta_0 and beta_1
sigma(slr_fit)             # Residual standard error (RSE)

# --- 2.4 Prediction at horsepower = 98 ---
# predict() returns the fitted value; interval = "confidence" gives CI for E[Y|X]
# and interval = "prediction" gives PI for an individual new observation.
new_data <- data.frame(horsepower = 98)

# Confidence interval: Where the AVERAGE mpg lies for all cars with hp = 98
predict(slr_fit, new_data, interval = "confidence")

# Prediction interval: Where a SINGLE new car's mpg would lie (always wider
# because it includes the irreducible error variance)
predict(slr_fit, new_data, interval = "prediction")

# --- 2.5 Plot the regression line ---
ggplot(Auto, aes(x = horsepower, y = mpg)) +
  geom_point(alpha = 0.4, color = "#1E2761") +
  geom_smooth(method = "lm", se = TRUE, color = "#3B82F6", fill = "#CADCFC") +
  labs(title = "Simple Linear Regression: mpg ~ horsepower",
       subtitle = paste0("R² = ", round(summary(slr_fit)$r.squared, 3),
                         ", RSE = ", round(sigma(slr_fit), 2)),
       x = "Horsepower", y = "MPG") +
  theme_minimal()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Diagnostic Plots — Checking Assumptions (Lines 85–115)
# ══════════════════════════════════════════════════════════════════════════════

# --- 3.1 The four standard diagnostic plots ---
# R's plot(lm_object) produces four diagnostic plots:
#   which=1: Residuals vs Fitted — checks linearity
#   which=2: Normal Q-Q — checks normality of residuals
#   which=3: Scale-Location — checks constant variance (homoscedasticity)
#   which=5: Residuals vs Leverage — identifies influential observations
par(mfrow = c(2, 2))
plot(slr_fit)
par(mfrow = c(1, 1))

# --- 3.2 Interpret the Residuals vs Fitted plot ---
# A clear U-shaped pattern in the residuals indicates non-linearity!
# The linear model misses the curvature in the mpg-horsepower relationship.
# Solution: Add a quadratic term (Section 5 below).
ggplot(data.frame(fitted = fitted(slr_fit), residuals = residuals(slr_fit)),
       aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.4, color = "#1E2761") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "#EF4444") +
  geom_smooth(se = FALSE, color = "#F59E0B", linewidth = 0.8) +
  labs(title = "Residuals vs. Fitted Values (SLR)",
       subtitle = "U-shape indicates non-linearity — consider polynomial term",
       x = "Fitted Values", y = "Residuals") +
  theme_minimal()

# --- 3.3 Studentized residuals ---
# Values > 3 in absolute value suggest potential outliers.
# rstudent() divides each residual by its estimated standard deviation,
# accounting for leverage.
sr <- rstudent(slr_fit)
cat("Max absolute studentized residual:", max(abs(sr)), "\n")
cat("Observations with |studentized residual| > 3:\n")
which(abs(sr) > 3)

# --- 3.4 Leverage statistics ---
# hatvalues() returns the diagonal of the hat matrix H = X(X'X)^-1 X'.
# Average leverage = (p+1)/n = 2/392 ≈ 0.005.
# Points with leverage far exceeding the average are potentially influential.
hv <- hatvalues(slr_fit)
avg_leverage <- 2 / nrow(Auto)  # (p+1)/n for SLR with one predictor
cat("Average leverage:", round(avg_leverage, 4), "\n")
cat("Observations with leverage > 3× average:\n")
which(hv > 3 * avg_leverage)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: Multiple Linear Regression (Lines 40–60 on slides)
# ══════════════════════════════════════════════════════════════════════════════

# --- 4.1 Fit MLR with all predictors except 'name' ---
# The dot notation (. - name) means "use all columns except name".
# This fits: mpg = beta_0 + beta_1*cylinders + beta_2*displacement + ... + epsilon
mlr_fit <- lm(mpg ~ . - name, data = Auto)
summary(mlr_fit)

# --- 4.2 Identify significant predictors ---
# Look at p-values in the Pr(>|t|) column.
# displacement, weight, year, and origin are significant (p < 0.05).
# horsepower is NOT significant here — it's collinear with displacement and weight.
# When both weight and displacement are in the model, they absorb horsepower's effect.

# --- 4.3 Year coefficient interpretation ---
# year ≈ 0.75: Each model year, mpg improves by ~0.75 on average,
# holding other variables constant. Reflects technological improvement over time.

# --- 4.4 R² comparison ---
# MLR R² ≈ 0.82 vs. SLR R² ≈ 0.61 — the additional predictors explain
# substantially more variance in mpg.
cat("SLR R²:", round(summary(slr_fit)$r.squared, 3), "\n")
cat("MLR R²:", round(summary(mlr_fit)$r.squared, 3), "\n")

# --- 4.5 Variance Inflation Factors ---
# vif() from the car package measures multicollinearity.
# VIF > 5 signals concern; VIF > 10 is severe.
# cylinders, displacement, and weight have high VIF → collinear.
vif(mlr_fit)

# --- 4.6 Diagnostic plots for MLR ---
par(mfrow = c(2, 2))
plot(mlr_fit)
par(mfrow = c(1, 1))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: Interaction Terms & Polynomial Regression (Lines 65–140)
# ══════════════════════════════════════════════════════════════════════════════

# --- 5.1 Interaction: displacement × weight ---
# The * operator creates main effects AND the interaction term.
# This tests whether the effect of displacement on mpg depends on weight.
int_fit <- lm(mpg ~ displacement * weight, data = Auto)
summary(int_fit)
# If the interaction term p-value < 0.05, the effect of displacement on mpg
# varies with weight level (synergy or antagonism between the two).

# --- 5.2 Polynomial regression: mpg ~ horsepower + horsepower² ---
# I() protects the arithmetic operator so R treats it as a mathematical
# transformation rather than a formula operator.
quad_fit <- lm(mpg ~ horsepower + I(horsepower^2), data = Auto)
summary(quad_fit)

# Compare: R² jumps from 0.606 (linear) to ~0.688 (quadratic)
# The quadratic term is highly significant (p < 0.001), confirming non-linearity.

# --- 5.3 Visualize the quadratic fit ---
ggplot(Auto, aes(x = horsepower, y = mpg)) +
  geom_point(alpha = 0.4, color = "#1E2761") +
  geom_smooth(method = "lm", formula = y ~ x, se = FALSE,
              color = "#EF4444", linetype = "dashed", linewidth = 0.8) +
  geom_smooth(method = "lm", formula = y ~ poly(x, 2), se = FALSE,
              color = "#10B981", linewidth = 0.8) +
  labs(title = "Linear (red dashed) vs. Quadratic (green) Fit",
       subtitle = "Quadratic model captures the curvature much better",
       x = "Horsepower", y = "MPG") +
  theme_minimal()

# --- 5.4 Compare models using ANOVA ---
# anova() performs an F-test comparing nested models.
# A significant p-value means the more complex model fits significantly better.
anova(slr_fit, quad_fit)

# --- 5.5 Using poly() for orthogonal polynomials ---
# poly(x, degree) creates orthogonal polynomial basis functions.
# This avoids collinearity between x, x², x³, etc.
poly_fit <- lm(mpg ~ poly(horsepower, 5), data = Auto)
summary(poly_fit)
# Degrees 1 and 2 are significant; degree 5 is not → quadratic is sufficient.

# --- 5.6 Log transformation ---
# log(horsepower) can also capture the diminishing-returns relationship.
log_fit <- lm(mpg ~ log(horsepower), data = Auto)
summary(log_fit)
# R² ≈ 0.67 — slightly less than quadratic but more interpretable:
# A 1% increase in horsepower → mpg changes by beta_1/100 units.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EXERCISES — Complete these for homework
# ══════════════════════════════════════════════════════════════════════════════

# ── Exercise 1: Simple Linear Regression (ISLR 3.7 Q8) ──────────────────
# Dataset: Auto | Response: mpg | Predictor: horsepower
# (a) Fit SLR, print summary
# (b) Interpret the coefficient of horsepower
# (c) Predict mpg at horsepower = 98
# (d) 95% CI and PI at horsepower = 98
# (e) Plot mpg vs horsepower with regression line
# (f) Produce diagnostic plots and comment

## YOUR CODE HERE ##


# ── Exercise 2: Multiple Linear Regression (ISLR 3.7 Q9) ────────────────
# Dataset: Auto | Response: mpg | Predictors: all except name
# (a) Scatterplot matrix: pairs(Auto %>% select(-name))
# (b) Correlation matrix
# (c) Fit MLR
# (d) Which predictors are significant?
# (e) What does the year coefficient suggest?
# (f) Diagnostic plots
# (g) Try interaction terms
# (h) Try transformations: log, sqrt, x^2

## YOUR CODE HERE ##


# ── Exercise 3: Carseats (ISLR 3.7 Q10) ─────────────────────────────────
# Dataset: Carseats | Response: Sales | Predictors: Price, Urban, US
# (a) Fit MLR: Sales ~ Price + Urban + US
# (b) Interpret each coefficient
# (c) Write the model in equation form
# (d) Reject H0 for which predictors?
# (e) Fit reduced model with significant predictors only
# (f) Compare models
# (g) 95% CIs for reduced model

## YOUR CODE HERE ##


# ── Exercise 4: Boston EDA (ISLR 2.4 Q10) ────────────────────────────────
# Dataset: Boston | Response: various
# (a) Dimensions and variable meanings
# (b) Pairwise scatterplots
# (c) Predictors associated with per capita crime rate
# (d) Tracts with unusually high crime, tax, ptratio
# (e) Count: tracts bounding Charles River
# (f) Median pupil-teacher ratio
# (g) Tract with lowest median home value — comment

## YOUR CODE HERE ##


# ══════════════════════════════════════════════════════════════════════════════
# SOLUTION KEYS
# ══════════════════════════════════════════════════════════════════════════════

# ── Solution: Exercise 1 ─────────────────────────────────────────────────

# (a)
ex1_fit <- lm(mpg ~ horsepower, data = Auto)
summary(ex1_fit)

# (b) Each unit increase in horsepower is associated with a decrease of
#     approximately 0.158 in mpg. The relationship is statistically significant
#     (p < 2e-16), and the model explains 60.6% of variance in mpg.

# (c)
predict(ex1_fit, data.frame(horsepower = 98))
# ≈ 24.47 mpg

# (d)
predict(ex1_fit, data.frame(horsepower = 98), interval = "confidence")
# 95% CI for E[Y|X=98]: approximately [23.97, 24.96]
predict(ex1_fit, data.frame(horsepower = 98), interval = "prediction")
# 95% PI for Y_new at X=98: approximately [14.81, 34.12]
# The PI is much wider because it includes irreducible error variance.

# (e)
ggplot(Auto, aes(x = horsepower, y = mpg)) +
  geom_point(alpha = 0.4, color = "#475569") +
  geom_smooth(method = "lm", se = TRUE, color = "#3B82F6", fill = "#CADCFC") +
  labs(title = "Exercise 1(e): mpg vs. horsepower with regression line",
       x = "Horsepower", y = "MPG") +
  theme_minimal()

# (f) Diagnostic plots
par(mfrow = c(2, 2))
plot(ex1_fit)
par(mfrow = c(1, 1))
# Residual plot shows U-shaped pattern → non-linearity.
# Q-Q plot: approximately normal with slight heavy tails.
# Observation 334: high leverage (very high horsepower).


# ── Solution: Exercise 2 ─────────────────────────────────────────────────

# (a)
pairs(Auto %>% select(-name), pch = 16, cex = 0.3)

# (b)
round(cor(Auto %>% select(-name)), 2)

# (c)
ex2_fit <- lm(mpg ~ . - name, data = Auto)
summary(ex2_fit)

# (d) Significant: displacement (p=0.009), weight (p<0.001), year (p<0.001),
#     origin (p<0.001). NOT significant: cylinders, horsepower, acceleration.

# (e) year coefficient ≈ 0.75: each additional model year improves mpg by
#     about 0.75, reflecting technological advances in fuel efficiency.

# (f)
par(mfrow = c(2, 2))
plot(ex2_fit)
par(mfrow = c(1, 1))
# Residual plot: some curvature remains, suggesting non-linear terms may help.
# Observation 14 has high leverage.

# (g) Interaction terms
ex2_int <- lm(mpg ~ displacement * weight + year + origin, data = Auto)
summary(ex2_int)
# displacement:weight interaction is significant (p < 0.001).

# (h) Transformations
ex2_log <- lm(mpg ~ log(horsepower) + log(weight) + year + origin, data = Auto)
summary(ex2_log)
# R² improves to ~0.86 — log transforms capture non-linearity well.


# ── Solution: Exercise 3 ─────────────────────────────────────────────────

# (a)
data(Carseats)
ex3_fit <- lm(Sales ~ Price + Urban + US, data = Carseats)
summary(ex3_fit)

# (b) Price: Each $1 increase in price → Sales decrease by ~0.054 units (thousand).
#     UrbanYes: Being in an urban location has no significant effect (p ≈ 0.94).
#     USYes: US-located stores have ~1.2 units higher sales than non-US stores.

# (c) Sales = 13.04 - 0.054*Price - 0.022*I(Urban=Yes) + 1.20*I(US=Yes) + epsilon

# (d) Reject H0 for Price (p < 0.001) and US (p < 0.001).
#     Cannot reject H0 for Urban (p = 0.936).

# (e) Reduced model
ex3_reduced <- lm(Sales ~ Price + US, data = Carseats)
summary(ex3_reduced)

# (f) Comparison
# R² barely changes (0.2393 vs 0.2354); Adjusted R² slightly improves
# in the reduced model because Urban adds no explanatory power.
cat("Full model Adj R²:", summary(ex3_fit)$adj.r.squared, "\n")
cat("Reduced model Adj R²:", summary(ex3_reduced)$adj.r.squared, "\n")

# (g) 95% CIs
confint(ex3_reduced)


# ── Solution: Exercise 4 ─────────────────────────────────────────────────

# (a)
data(Boston)
dim(Boston)   # 506 rows (census tracts), 13 columns
?Boston       # Documentation with variable descriptions

# (b)
pairs(Boston[, c("crim", "zn", "indus", "nox", "rm", "age", "dis", "medv")],
      pch = 16, cex = 0.3)

# (c) Predictors associated with crime rate:
round(cor(Boston)[, "crim"], 2)
# Strong associations: rad (0.63), tax (0.58), lstat (0.46), nox (0.42)
# These reflect socioeconomic and infrastructure patterns.

# (d) Tracts with extreme values
cat("High crime (>20):", sum(Boston$crim > 20), "tracts\n")
cat("High tax (>600):", sum(Boston$tax > 600), "tracts\n")
cat("Low ptratio (<14):", sum(Boston$ptratio < 14), "tracts\n")
# These are outliers; most tracts cluster at lower values.

# (e) Charles River
sum(Boston$chas == 1)   # 35 tracts bound the Charles River

# (f) Median pupil-teacher ratio
median(Boston$ptratio)  # ≈ 19.05

# (g) Tract with lowest median home value
worst <- which.min(Boston$medv)
cat("Tract with lowest medv:", worst, "\n")
Boston[worst, ]
# This tract has: very high crime, 100% old housing (age=100),
# high lstat (low socioeconomic status), high tax — the most
# disadvantaged census tract in the dataset.


# ══════════════════════════════════════════════════════════════════════════════
# END OF LAB_WEEK1.R
# ══════════════════════════════════════════════════════════════════════════════
