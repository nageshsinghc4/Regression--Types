#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:29:33 2019

@author: nageshsinghchauhan
"""

import numpy
import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/MachineLearning/Part_2-Regression/Section_4_Simple_Linear_Regression/Salary_Data.csv')

#X -> matrix of independent variable
#y -> vector of dependent variable
X = data.iloc[:,:-1].values
y = data.iloc[:,1].values

#splitting the dataset into test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.30, random_state= 0)

#No need for feature scaling
#fitting simple linera regression model to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)
    
#visualize the results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train))