########################### First part ###########################

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

#import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features
y = iris.target

#we create an instance of SVM and fit out data. We do not scale our 
#data since we want to plot the support vectors
model1 = svm.SVC(kernel='rbf', C=1).fit(X, y)

#create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

#plot the countour plot
plt.subplot(1, 1, 1)
Z = model1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)

#plot points and define axis labels
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=50)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

########################### Second part ###########################

#import module for cross validation
from sklearn.model_selection import KFold

#set simple K-Fold cross validation object - 10 folds
kf = KFold(shuffle=True, n_splits=10)

results_total_test = []
C_values = np.logspace(-5,5,20,base=10)
for C in C_values:
    #create the SVC object
    model2 = svm.SVC(kernel='rbf', C=C)
    
    #obtain misclassification rate for each cross validation set
    results_test = []
    for train_index, test_index in kf.split(X):
            prediction_test = model2.fit(X[train_index], y[train_index]).predict(X[test_index])
            results_test.append(1-(sum(prediction_test==y[test_index]))/len(test_index))
    
    res_test = np.array(results_test).mean()
    results_total_test.append(res_test)
    #print ("Misclassification Rate on test set: " + str(res_test))

#plot final results             
plt.plot(np.log10(C_values), results_total_test, 'go-', label='Test', linewidth=2)
plt.title('Misclassification Rate')
plt.legend()

            
            