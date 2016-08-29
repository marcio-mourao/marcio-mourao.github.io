#Load the Boston dataset from datasets
#This dataset contains 13 housing variables measured on 506 houses,
#with house price as the outcome. 

from sklearn import datasets
import numpy as np
boston = datasets.load_boston()

#Define the training and test datasets
#The training set consists of the first 50% of the data (221 observations)
#and the test set consists of the remaining 221 observations.
boston_X_train = boston.data[:-221]
boston_X_test = boston.data[-221:]
boston_y_train = boston.target[:-221]
boston_y_test = boston.target[-221:]
pricy_train = np.less(30,boston_y_train)
pricy_test = np.less(30,boston_y_test)


#Fit logistic regression models to boston_X_train and pricy_train for all
#penalties. Evaluate the models on the test set.
from sklearn import linear_model, metrics
regr = linear_model.LogisticRegression()
penalties = [0.01,0.1,1,10,100]
accuracy = []
f_measure = []
precision = []
recall = []
for penalty in penalties:
    regr.set_params(C=penalty).fit(boston_X_train,pricy_train)
    pred_test = regr.predict(boston_X_test)
    accuracy.append(metrics.accuracy_score(pricy_test,pred_test))
    f_measure.append(metrics.f1_score(pricy_test,pred_test))
    precision.append(metrics.precision_score(pricy_test,pred_test))
    recall.append(metrics.recall_score(pricy_test,pred_test))                  
print(accuracy)
print(f_measure)
print(precision)
print(recall)
