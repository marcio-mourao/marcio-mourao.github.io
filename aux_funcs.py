import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Just displays the Iris dataset in a tridimensional setting
def display_Iris(X,fignum,labels):
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    
    for name, label in [('Setosa', 0),
                        ('Versicolour', 1),
                        ('Virginica', 2)]:                            
                            ax.text3D(X[labels == label, 3].mean(),
                                      X[labels == label, 0].mean() + 1.5,
                                      X[labels == label, 2].mean(), name,
                                      horizontalalignment='center',
                                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

#Defines clusters based on the Guassian distribution
def init_board_gauss(N, k, s2):
    n = float(N)/k
    X = []
    target = []
    for i in range(k):
        c = (random.uniform(-1, 1), random.uniform(-1, 1))
        s = random.uniform(s2,s2)
        x = []        
        while len(x) < n:
            a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
            # Continue drawing points from the distribution in the range [-1,1]
            if abs(a) < 1 and abs(b) < 1:
                x.append([a,b])
                target.append(i)
        X.extend(x)
    X = np.array(X)[:N]
    target=np.array(target)
    return [X,target]
