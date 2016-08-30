import random
import aux_funcs
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

#Initializes the seed for the random numbers
np.random.seed(100)

# Generate points for the exercise
nPoints=800
nClasses=10
s2=0.1
[X,target]=aux_funcs.init_board_gauss(nPoints,nClasses,s2)
points=np.array([[random.uniform(-1, 1),random.uniform(-1, 1)],
                 [random.uniform(-1, 1),random.uniform(-1, 1)]])

#set the model
model = KNeighborsClassifier(5,weights='uniform')
#model = GaussianNB()
#model = DecisionTreeClassifier()

model.fit(X, target)
predictions=model.predict(points)
X=np.concatenate((X,points),axis=0)
target=np.concatenate((target,predictions),axis=0)
plt.figure(4,figsize=(10,10))
plt.scatter(X[:,0],X[:,1],c=target,s=nPoints*[100])
plt.colorbar()
plt.scatter(points[:,0],points[:,1],facecolors='none',edgecolors='k',marker='s',s=(points.shape[0])*[300])
plt.show()

