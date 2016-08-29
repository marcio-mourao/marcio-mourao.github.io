#Load the Boston dataset from datasets
#This dataset contains 13 housing variables measured on 506 houses,
#with house price as the outcome. 

from sklearn import datasets
import numpy as np
boston = datasets.load_boston()

#Define the training and test datasets
#The training set consists of the first 50% of the data (253 observations)
#and the test set consists of the remaining 253 observations.
boston_X_train = boston.data[:-253]
boston_X_test = boston.data[-253:]
boston_y_train = boston.target[:-253]
boston_y_test = boston.target[-253:]
pricy_train = np.less(30,boston_y_train)
pricy_test = np.less(30,boston_y_test)

#Fit a logistic regression model to boston_X_train and pricy_train
from sklearn import linear_model
regr = linear_model.LogisticRegression()
regr.fit(boston_X_train, pricy_train)

#Assess the accuracy of the model on pricy_test
from sklearn import metrics
test_pred = regr.predict(boston_X_test)
print(metrics.accuracy_score(pricy_test,test_pred))
#Create confusion matrix for the model
print(metrics.confusion_matrix(test_pred,pricy_test))
#Assess precision, recall, and F-measure for the model
print(metrics.precision_score(test_pred,pricy_test))
print(metrics.recall_score(test_pred,pricy_test))
print(metrics.f1_score(test_pred,pricy_test))

#Get the probabilities, calculate an AUC value, and graph an ROC curve
test_prob_yes = regr.predict_proba(boston_X_test)[:,1]
print(metrics.roc_auc_score(pricy_test,test_prob_yes))
roc=metrics.roc_curve(pricy_test,test_prob_yes)
import matplotlib.pyplot as plt
#Sets up the ROC plot
fpr = roc[0]
tpr = roc[1]
plt.scatter(fpr,tpr)
#Connects the points in the ROC plot with lines
plt.plot(fpr,tpr)
#Adds diagonal line
plt.plot([0,1],[0,1])
#Sets up the axis labels
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.xlim(0,1.01)
plt.ylim(0,1.01)
#Displays the plot
plt.show()

#Assess the calibration of the model by making a table of predicted probability
#decile versus actual probability.
def decile(x): 
    return int(10*x)
#Turns decile into a function that I can apply on an array
decile = np.vectorize(decile)
test_prob_decile = decile(test_prob_yes)
#Pandas implements R functionality into Python
import pandas as pd
df = pd.DataFrame({'Decile':test_prob_decile, 'Actual':pricy_test})
#Frequency and average probability grouped by decile
print(df.groupby('Decile').count())
print(df.groupby('Decile').mean())