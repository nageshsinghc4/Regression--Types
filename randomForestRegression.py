#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 16:45:39 2019
@author: nageshsinghchauhan
"""
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/MachineLearning/Part_2-Regression/Section9-RandomForestRegression/Position_Salaries.csv')

#X -> matrix of independent variable
#y -> vector of dependent variable
X = data.iloc[:,1:2].values
y = data.iloc[:,2].values

#splitting the dataset into test and training data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0)


"""
#feature scaling
#not used that much because all ML lib already include it
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))
"""

#fitting Random forst model to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,y)

y_pred = regressor.predict(6.5)

#visualize Decision tree output in high resolution and moother curves
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()