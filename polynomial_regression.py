# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import dataset
dataset = pd.read_csv('claim_per_policy.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Visualise dataset
plt.scatter(X, y, color = 'red')
plt.title('Claim per Policy')
plt.xlabel('Age')
plt.ylabel('Claim Amount')
plt.show()


# Simple Linear Regression for entire dataset (For comparison)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)


# Polynomial Regression for entire dataset (polynomial fit followed by Linear regression)
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree = 4)
X_poly = poly_features.fit_transform(X)
poly_features.fit(X_poly, y)
poly_regressor = LinearRegression()
poly_regressor.fit(X_poly, y)


# Predict with Linear Regression for Age: 57
lr_pred = linear_regressor.predict(np.array([[57]]))

print("Linear prediction for Age: 57")
print(lr_pred)

# Predict with Polynomial Regression for Age: 57
poly_pred = poly_regressor.predict(poly_features.fit_transform(np.array([[57]])))

print("Polynomial prediction for Age: 57")
print(poly_pred)


# Visualise Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Claim per Policy (Linear Regression)')
plt.xlabel('Age')
plt.ylabel('Claim Amount')
plt.show()


# Visualise Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, poly_regressor.predict(poly_features.fit_transform(X)), color = 'blue')
plt.title('Claim per Policy (Polynomial Regression)')
plt.xlabel('Age')
plt.ylabel('Claim Amount')
plt.show()
