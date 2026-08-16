# Telecom Customer Churn Prediction

Which telecom customers are about to leave, and which attributes predict it?

## Question

Given a customer's contract, billing, and service-usage attributes, can we predict churn
well enough to rank customers by risk and point retention spend at the top of that ranking?

## Data

IBM Telco Customer Churn (Kaggle, `blastchar/telco-customer-churn`): 7,043 customers,
21 columns. `TotalCharges` arrives as a string with 11 blank entries, which are coerced to
NaN and dropped, leaving 7,032 rows. Class balance: 5,163 retained vs 1,869 churned (26.6%).

## Approach

**Features.** `tenure`, `MonthlyCharges` and `TotalCharges` are correlated, so rather than
drop one, all three were standardized and reduced with PCA: the first two components hold
98.0% of their variance and carry forward as PC1 and PC2. Chi-square tests of independence
on every categorical field against churn kept only the significant ones. One-hot encoding
used m-1 dummies. Final feature set: 13 columns.

**Modeling.** A 90/10 split carves out a 704-customer test set that stays untouched until
the end, then 80/20 within the remainder for validation. SMOTE is applied to the training
fold only. Six models tuned with `RandomizedSearchCV` (50 iterations, 5-fold, scored on F1).

## Results

Cross-validated F1 on the resampled training data, with validation accuracy:

| Model | CV F1 | Validation accuracy |
|---|---|---|
| Gradient Boosting | 0.821 | 0.765 |
| XGBoost | 0.811 | 0.766 |
| Random Forest | 0.810 | 0.774 |
| AdaBoost | 0.787 | 0.765 |
| Logistic Regression | 0.771 | 0.770 |
| Neural Network | 0.729 | 0.752 |

Logistic regression was carried to the held-out test set: accuracy 0.714, F1 0.564,
recall on churners 0.722, specificity 0.712, precision 0.463. Confusion matrix: 373 TN,
151 FP, 50 FN, 130 TP.

It catches 72% of actual churners at 46% precision. For a retention campaign that is usually
the right trade, since a wasted contact costs far less than a silent churn. Logistic
regression also keeps the coefficients readable, and the notebook closes with feature
importance plus cumulative gains and decile lift curves.

## Running it

Open the notebook in Jupyter or Colab. The dataset is public on Kaggle. Requires pandas,
scikit-learn, imbalanced-learn, xgboost, scikit-plot. Kaggle credentials are read from
environment variables or `~/.kaggle/kaggle.json`, not hardcoded.

## Context

Graduate coursework, IE 7275 Data Mining in Engineering, Northeastern University, spring
2023. Two-person team project, effort recorded as 50/50 in the project report. Published
for reference: a course assignment, not production code.
