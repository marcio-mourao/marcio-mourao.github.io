#Load the diabetes dataset from datasets
#This dataset contains 10 physiological variables measured on 442 patients,
#with an outcome that measures disease progression after one year. 
#You can use np.loadtxt to read in a test file into a numpy matrix that
#you can use to define the data and target to fit a model.
from sklearn import datasets
diabetes = datasets.load_diabetes()

#Define the training and test datasets
#The training set consists of the first 50% of the data (221 observations)
#and the test set consists of the remaining 221 observations.
diabetes_X_train = diabetes.data[:-221]
diabetes_X_test = diabetes.data[-221:]
diabetes_y_train = diabetes.target[:-221]
diabetes_y_test = diabetes.target[-221:]

#Fit a linear regression model to diabetes_X_test and diabetes_y_test
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print(regr.coef_)

#Calculate residuals and mean squared error
import numpy as np
residuals = regr.predict(diabetes_X_train)-diabetes_y_train
print(np.mean(residuals**2))

#Calculate the percentage of variance explained
print(regr.score(diabetes_X_train,diabetes_y_train))

#Evaluate model performance on test set
print(np.mean((regr.predict(diabetes_X_test)-diabetes_y_test)**2))
print(regr.score(diabetes_X_test,diabetes_y_test))

#Create QQ plot of residuals
import matplotlib.pyplot as plt
import scipy.stats as stats
import pylab
stats.probplot(residuals, dist="norm", plot=pylab)
#Displays the plot. py.savefig("filename.type") will save to a file of 
#format type.
pylab.show()

#Create scatterplot of residuals by outcome
plt.scatter(diabetes_y_train,residuals)
plt.xlabel('Diabetes progress in last year')
plt.ylabel('Residual')
#Displays the scatterplot. plt.savefig("filename.type") will save the plot to
#a file of format type.
plt.show()

#Optimize Ridge regression model fit on training data based on test data
#The test values for alpha will be (10**(-7), 10**(-6),..., 10**(-1),1)
alpha_test = [0.00000001, 0.0000001, 0.00001,0.0001,0.001,0.01,0.1,1]
regr_ridge = linear_model.Ridge()
#Gives the scores on the test data for a model 
#fit on the training data with the given alpha
scores_ridge = [regr_ridge.set_params(alpha=alpha)
                          .fit(diabetes_X_train,diabetes_y_train)
                          .score(diabetes_X_test,diabetes_y_test)
                for alpha in alpha_test]
print(scores_ridge)

#Creates a figure displaying the results of the optimization
#The first part sets up the log scale on the y-axis
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
#First, find the alpha giving the best model
alpha_best = alpha_test[scores_ridge.index(max(scores_ridge))]
print(alpha_best)
#Fit a Ridge regression model with that alpha
regr_ridge.set_params(alpha=alpha_best).fit(diabetes_X_train,diabetes_y_train)
print(regr_ridge.coef_)

#Optimize Lasso regression
regr_lasso = linear_model.Lasso()
scores_lasso = [regr_lasso.set_params(alpha=alpha, max_iter=10000)
                          .fit(diabetes_X_train,diabetes_y_train)
                          .score(diabetes_X_test,diabetes_y_test)
                for alpha in alpha_test]
print(scores_lasso)
alpha_best=alpha_test[scores_lasso.index(max(scores_lasso))]
print(alpha_best)
regr_lasso.set_params(alpha=alpha_best).fit(diabetes_X_train,diabetes_y_train)
print(regr_lasso.coef_)