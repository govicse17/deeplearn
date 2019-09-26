# Simple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('auto_insurance_payment.csv')
dataset.describe()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Visualise dataset
plt.scatter(X, y, color = 'red')
plt.title('Swedish Automobile Insurance')
plt.xlabel('Number of Claims')
plt.ylabel('Total payment')
plt.show()


# Split dataset into Training and Test set
# sklearn.cross_validation  - Deprecated
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
# Alternative: ColumnTransformer
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = sc_y.fit_transform(np.array(y_train).reshape(-1, 1))
X_test = sc_X.fit_transform(X_test)
y_test_org = y_test
y_test = sc_y.fit_transform(np.array(y_test).reshape(-1, 1))


# Simple Linear Regression with Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict Test set
y_pred = regressor.predict(X_test)

# Print unscaled test and predicted values
y_pred_inv = sc_y.inverse_transform(y_pred)
print(pd.DataFrame(np.column_stack((y_test_org, y_pred_inv))))


# Print Coefficient & Intercept
print('Coefficient: ', regressor.coef_)
print('Intercept: ', regressor.intercept_)

# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, accuracy_score
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % explained_variance_score(y_test, y_pred))
# Coefficient of determination
print('R^2 Square value', r2_score(y_test, y_pred))



# Visualise Training set result
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Swedish Automobile Insurance Data (Training set)')
plt.xlabel('Number of Claims')
plt.ylabel('Total payment')
plt.show()

# Visualise Test set result
plt.scatter(X_test, y_test, color = 'green')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Swedish Automobile Insurance Data (Test set)')
plt.xlabel('Number of Claims')
plt.ylabel('Total payment')
plt.show()
