# Random Forest Regression #2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('health_insurance_cost.csv')
dataset.describe()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values


# Preprocessing: 
# Handle Categorical variables using OneHotEncoder
# Preprocess Gender column
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
colT = ColumnTransformer(remainder='drop', transformers=
    [("dummy_gender", OneHotEncoder(categories='auto'), [1])])
genders = colT.fit_transform(X)
# Avoid Dummy variable trap
genders = genders[:, 1:]

# Preprocess Smoker column
colT = ColumnTransformer(remainder='drop', transformers=
    [("dummy_smoker", OneHotEncoder(categories='auto'), [4])])
smokers = colT.fit_transform(X)
# Avoid Dummy variable trap
smokers = smokers[:, 1:]


# Preprocess Region column
colT = ColumnTransformer(remainder='drop', transformers=
    [("dummy_region", OneHotEncoder(categories='auto'), [5])])
regions = colT.fit_transform(X)
# Avoid Dummy variable trap
regions = regions[:, 1:]


# Remove the original categorial columns
X = np.delete(X, [1,4,5], axis=1)

# Concatenate dummy variables
X = np.concatenate((genders, smokers, regions.toarray(), X), axis=1)


# Split dataset into Training and Test set
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


# Random Forest Regression for training set ( 10 estimators )
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train.ravel())

# Predict Test set using Random Forest regressor
y_pred = regressor.predict(X_test)

# Print unscaled test and predicted values
y_pred_inv = sc_y.inverse_transform(y_pred)
print(pd.DataFrame(np.column_stack((y_test_org, y_pred_inv))).head(10))


print('--------------------------------------------')
# Metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
print("Mean absolute error: %.2f" % mean_absolute_error(y_test, y_pred))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))
print("Root Mean squared error: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % explained_variance_score(y_test, y_pred))
# Coefficient of determination
print('R^2 Square value', r2_score(y_test, y_pred))
