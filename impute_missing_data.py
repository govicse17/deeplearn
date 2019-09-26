# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('health_insurance_cost_missing_data.csv')
dataset.describe()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

# Fill missing numerical data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, [0,2,3]])
X[:, [0,2,3]] = imputer.transform(X[:, [0,2,3]])

# Fill missing categorical data
imputer = SimpleImputer(strategy="most_frequent")
imputer = imputer.fit(X[:, [1,4,5]])
X[:, [1,4,5]] = imputer.transform(X[:, [1,4,5]])
print(X)