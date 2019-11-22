#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:43:51 2019

@author: nageshsinghchauhan
"""


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv('/Users/nageshsinghchauhan/Downloads/ML/MachineLearning/Part_2-Regression/Section_6_Polynomial_Regression/Position_Salaries.csv')

#X -> matrix of independent variable
#y -> vector of dependent variable
X = data.iloc[:,1:2].values
y = data.iloc[:,2].values

"""
#splitting the dataset into test and training data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 0)
"""

#fitting linear regressor model in the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#fitting Polynomial linear regressor model in the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualize Linear regression model 
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')

#visualize Polynomial regression model
plt.scatter(X,y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')

#predict the new result with Linear regression
lin_reg.predict(7)


#predict the new result with Ploynomial Linear regression
lin_reg.predict(poly_reg.fit_transform(6))