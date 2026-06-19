# Regression & Classification

Worked examples for **Chapter 2 (Fundamental Machine Learning)**, accompanying the
linear- and logistic-regression subsections. Linear regression fits $y = wx + b$ to
noisy 1D data; logistic regression learns a decision boundary between two classes in
2D. The `_sklearn` variants reproduce the same problems with polynomial features
through scikit-learn pipelines.

## Drivers

- `linear_regression.py`
  steepest-descent fit of a line, cross-checked against the closed-form normal equations
- `linear_regression_outlier.py`
  one outlier under MSE, MAE, and the Huber loss (select with `CASE`)
- `linear_regression_sklearn.py`
  polynomial regression via a scikit-learn pipeline (degree `P`)
- `logistic_regression.py`
  steepest-descent binary classifier with a linear decision boundary
- `logistic_regres_sklearn.py`
  polynomial logistic regression; contour of the learned decision region (degree `P`)
