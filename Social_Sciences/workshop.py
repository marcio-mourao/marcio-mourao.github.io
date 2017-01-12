#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:50:40 2016

@author: mdam
"""

import numpy as np

##############################################################################
########################### One Dimensional Arrays ###########################
##############################################################################

##### Creation #####

a = np.array([1,2,3,4,5])
a.size
a.ndim
a.shape

b1 = np.arange(3)
b2 = np.arange(3,7)
b3 = np.arange(3,7,2)

c1 = np.zeros(5)
c2 = np.ones(3)
c3 = np.linspace(2.0, 3.0)
c4 = np.linspace(2,3,10)

c5 = np.random.rand(5)
c6 = np.random.randn(5)
c7 = 2.5 * np.random.randn(5) + 3

##### Indexing and Slicing #####

d = np.linspace(10,100,10)

d[0]
d[2]
d[-1]
d[-3]

d[2:7]
d[2:7:2]
d[3:]
d[:3]
d[:-3]
d[::2]
d[::-1]

d[2]
d[2]=15

e = np.linspace(1,5,5)
np.append(e,[6,7])
np.insert(e,2,[0.2,0.3,.4])
np.delete(e,[2,3])

e[[1,2,3,2,3,2,3]]
e[[1,2]]=[20,30]
e>5
e[e>5]

#Exercise 1 - Modify e to contain elements greater than 4 and elements less than 30
e=e[(e>4) & (e<30)]

##############################################################################
########################### Multidimensional Arrays ##########################
##############################################################################

##### Creation #####

g1 = np.zeros((2,3))
g2 = np.ones((2,3))

h = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
h.size
h.ndim
h.shape

##### Indexing and Slicing #####

h[0]
h[1]
h[1,2]
h[0:,1:]
h[:,0::2]
h[:,1]=[20,30,40,50]

aux=np.array([[1,2,3]])
np.append(h,aux)
np.append(h,aux,axis=0)
np.concatenate((h,aux),axis=0)

aux=np.array([[1,2,3,4]])
np.append(h,aux.T,axis=1)
np.concatenate((h,aux.T),axis=1)

h>15
aux2=h[h>15]

#Exercise 2 - Retrieve from h a bidimensional array with the elements 4,7,6,9
h[1:3,0:3:2]

##############################################################################
########################### Pandas dataframe  ################################
##############################################################################

import pandas as pd

data = pd.read_csv('Advertising.csv',index_col=0)
data.shape

data.isnull().any()
data.isnull().sum()

data[:5]
data.head()
data[::-1].head()
data[::-1].head().values

data.iloc[2:4,:]
data.loc[2:4,:]
data.loc[2:4,'TV']
data.loc[2:4,['TV','Sales']]

data['Sales']<10
data.loc[data['Sales']<10,:]

import matplotlib.pyplot as plt

fig1=plt.figure()
data.hist('TV')

#Visualize the relationship between the features and the response using scatterplots
data.plot(kind='scatter', x='TV', y='Sales', figsize=(4, 4))
data.plot(kind='scatter', x='Radio', y='Sales', figsize=(4, 4))
data.plot(kind='scatter', x='Newspaper', y='Sales', figsize=(4, 4))

##############################################################################
########################### Linear Regression  ###############################
##############################################################################

from sklearn import linear_model

#create linear regression object
reg = linear_model.LinearRegression()

##### Stage 1 #####

#set training X and target data y - starting with just TV as a covariate
X = np.array(data[['TV']]).reshape(-1,1)
y = np.array(data['Sales']).reshape(-1,1)

#train the model
reg.fit(X,y)

#print intercept & coefficients
print('\nIntercept: ', reg.intercept_)
print('\nCoefficients: ', reg.coef_)

#the mean squared error
print("\nMean squared error: %.2f" % np.mean((reg.predict(X) - y) ** 2))

#explained variance score: 1 is perfect prediction
print('\nExplained Fraction of Variance: %.2f ' % reg.score(X, y))

# Plot outputs
plt.figure()
plt.scatter(X, y, color='blue')
plt.plot(X, reg.predict(X), color='red',linewidth=3)
plt.xlabel('TV')
plt.ylabel('Sales')
plt.legend(['Fit','Raw Data'])

##### Stage 2 #####

#set training X and target data y - starting with all covariates
X = np.array(data[['TV','Radio','Newspaper']]).reshape(-1,3)
y = np.array(data['Sales']).reshape(-1,1)

#train the model
reg.fit(X,y)

#print intercept & coefficients
print('\n\n\n\nIntercept: ', reg.intercept_)
print('\nCoefficients: ', reg.coef_)

#the mean squared error
print("\nMean squared error: %.2f" % np.mean((reg.predict(X) - y) ** 2))

#explained variance score: 1 is perfect prediction
print('\nExplained Fraction of Variance: %.2f ' % reg.score(X, y))

#Preferred way
import statsmodels.formula.api as smf

results = smf.ols('Sales ~ TV + Radio + TV*Radio',data=data).fit()
print(results.summary())




