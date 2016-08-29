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

#Fit a linear SVM to pricy test and pricy train
from sklearn import svm
svc = svm.SVC(kernel = 'linear', C = 0.01)
svc.fit(boston_X_train, pricy_train)
#Output the coefficients of the SVM
print(svc.coef_)

#Assess the accuracy of the model on pricy_test
from sklearn import metrics
test_pred = svc.predict(boston_X_test)
print(metrics.accuracy_score(pricy_test,test_pred))
#Create confusion matrix for the model
print(metrics.confusion_matrix(test_pred,pricy_test))
#Assess precision, recall, and F-measure for the model
print(metrics.precision_score(test_pred,pricy_test))
print(metrics.recall_score(test_pred,pricy_test))
print(metrics.f1_score(test_pred,pricy_test))
