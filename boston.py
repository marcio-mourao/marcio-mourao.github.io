#Load the Boston dataset from datasets
#This dataset contains 13 housing variables measured on 506 houses,
#with house price as the outcome. 

from sklearn import datasets
boston = datasets.load_boston()

#Define the training and test datasets
#The training set consists of the first 50% of the data (506 observations)
#and the test set consists of the remaining 253 observations.
boston_X_train = boston.data[:-253]
boston_X_test = boston.data[-253:]
boston_y_train = boston.target[:-253]
boston_y_test = boston.target[-253:]

#Fit a linear regression model to boston_X_test and boston_y_test
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(boston_X_train, boston_y_train)
print(regr.coef_)

#Calculate residuals and mean squared error
import numpy as np
residuals = regr.predict(boston_X_train)-boston_y_train
print(np.mean(residuals**2))

#Calculate the percentage of variance explained
print(regr.score(boston_X_train,boston_y_train))

#Evaluate model performance on test set
print(np.mean((regr.predict(boston_X_test)-boston_y_test)**2))
print(regr.score(boston_X_test,boston_y_test))

#Optimize Ridge regression model fit on training data based on test data
alpha_test = np.logspace(0,7,15)
regr_ridge = linear_model.Ridge()
#Gives the scores on the test data for a model 
#fit on the training data with the given alpha
scores_ridge = [regr_ridge.set_params(alpha=alpha)
                          .fit(boston_X_train,boston_y_train)
                          .score(boston_X_test,boston_y_test)
                for alpha in alpha_test]
print(scores_ridge)

#Creates a figure displaying the results of the optimization
#The first part sets up the log scale on the y-axis
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1) #one row, one column, and one plot
ax.set_xscale('log')
#Sets up the scatter plot
plt.scatter(alpha_test,scores_ridge)
#Connects the points in the scatter plot with lines
plt.plot(alpha_test,scores_ridge)
#Sets up the axis labels
plt.xlabel('Alpha')
plt.ylabel('Percentage of variance explained on test set')
#Displays the plot
plt.show()

#Gives the coefficients from the best ridge regression model
#First, find the alpha giving the best model and the score for that model
alpha_best = alpha_test[scores_ridge.index(max(scores_ridge))]
print(alpha_best)
print(scores_ridge[scores_ridge.index(max(scores_ridge))])
#Fit a Ridge regression model with that alpha
regr_ridge.set_params(alpha=alpha_best).fit(boston_X_train,boston_y_train)
print(regr_ridge.coef_)

#Optimize Lasso regression
alpha_test = np.logspace(0,2,100)
regr_lasso = linear_model.Lasso()
scores_lasso = [regr_lasso.set_params(alpha=alpha, max_iter=10000)
                          .fit(boston_X_train,boston_y_train)
                          .score(boston_X_test,boston_y_test)
                for alpha in alpha_test]
                    
#Creates a figure displaying the results of the optimization
#The first part sets up the log scale on the y-axis
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1) #one row, one column, and one plot
ax.set_xscale('log')
#Sets up the scatter plot
plt.scatter(alpha_test,scores_lasso)
#Connects the points in the scatter plot with lines
plt.plot(alpha_test,scores_lasso)
#Sets up the axis labels
plt.xlabel('Alpha')
plt.ylabel('Percentage of variance explained on test set')
#Displays the plot
plt.show()      

#Displays the best alpha, the score, and the model             
alpha_best=alpha_test[scores_lasso.index(max(scores_lasso))]
print(alpha_best)
print(scores_lasso[scores_lasso.index(max(scores_lasso))])
regr_lasso.set_params(alpha=alpha_best).fit(boston_X_train,boston_y_train)
print(regr_lasso.coef_)