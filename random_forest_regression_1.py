# Random Forest Regression #1

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('claim_per_policy.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Simple Linear Regression for entire dataset (For comparison)
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)


# Random Forest Regression for entire dataset (with 10 estimators)
from sklearn.ensemble import RandomForestRegressor
randomforest_regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
randomforest_regressor.fit(X, y)


# Predicting a new result with Linear Regression
lin_reg_pred = linear_regressor.predict(np.array([[57]]))

print("Linear prediction for Age: 57")
print(lin_reg_pred)

# Predict using Random Forest Regressor for Age: 57
y_pred = randomforest_regressor.predict([[57]])
print("Random Forest prediction for Age: 57")
print(y_pred)


# Visualise Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Claim per Policy (Linear Regression)')
plt.xlabel('Age')
plt.ylabel('Claim Amount')
plt.show()

# Visualise Random Forest Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, randomforest_regressor.predict(X), color = 'blue')
plt.title('Claim per Policy (Random Forest Regression)')
plt.xlabel('Age')
plt.ylabel('Claim Amount')
plt.show()