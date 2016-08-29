import numpy as np

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#Initializes the seed for the random numbers
np.random.seed(100)

#Load the dataset
iris = datasets.load_iris()
X = iris.data

#Neighbours Classifier
X = iris.data
y = iris.target
n_neighbors=30
model = KNeighborsClassifier(n_neighbors, weights='uniform')
model.fit(X, y)
predictions=model.predict(X)
print("Number of mislabeled points out of a total %d points : %d" %(X.shape[0],(y != predictions).sum()))

#Naive Bayes Classifier
X = iris.data
y = iris.target
model = GaussianNB()
model.fit(X,y)
predictions=model.predict(X)
print("Number of mislabeled points out of a total %d points : %d" %(X.shape[0],(y != predictions).sum()))

#Decision Tree Classifier
X = iris.data
y = iris.target
model = DecisionTreeClassifier()
model.fit(X,y)
predictions=model.predict(X)
print("Number of mislabeled points out of a total %d points : %d" %(X.shape[0],(y != predictions).sum()))