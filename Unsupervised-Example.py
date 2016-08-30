import aux_funcs
import numpy as np

from sklearn import datasets
from sklearn.cluster import KMeans

#Initializes the seed for the random numbers
np.random.seed(100)

#Load the dataset
iris = datasets.load_iris()
X = iris.data

#Set three different KMeans Classifiers
model1=KMeans(n_clusters=3, n_init=10, init='k-means++')
model2=KMeans(n_clusters=8, n_init=10, init='k-means++')
model3=KMeans(n_clusters=3, n_init=1, init='random')

#Select the model, fit and display outcomes
model=model2
model.fit(X)
aux_funcs.display_Iris(X,1,model.labels_)
aux_funcs.display_Iris(X,2,iris.target)



