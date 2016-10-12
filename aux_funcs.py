import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Plot iris data points using the principal components as axis
def display_Iris_PCA(data,labels):
    plt.figure()
    
    if (len(labels)>1):
        names = ['setosa', 'versicolor', 'virginica']
        colors = ['navy', 'turquoise', 'darkorange']
        for classID, classLABEL, color in zip([0, 1 ,2], names, colors):
            plt.scatter(data[labels == classID, 0], 
                        data[labels == classID, 1], 
                        color=color, alpha=.8, lw=2, label=classLABEL)
    else:
        plt.scatter(data[:,0], 
                    data[:,1], 
                    color='black', alpha=.8, lw=2, label='')

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='best', shadow=True, scatterpoints=1)
    plt.title('PCA of IRIS dataset')

# Plot distribution of the different classes for each attribute
def display_Iris_Dist(data,labels):
    feature_dict = {0: 'sepal length [cm]',
                    1: 'sepal width [cm]',
                    2: 'petal length [cm]',
                    3: 'petal width [cm]'}

    plt.figure(figsize=(8, 6))
    for cnt in range(4):        
        plt.subplot(2,2,cnt+1)
        if (len(labels)>1):
            for classID,classLABEL in zip([0, 1, 2],['setosa', 'versicolor', 'virginica']):
                plt.hist(data[labels==classID, cnt], 
                         label=classLABEL, bins=10, alpha=0.3)
        else:
            plt.hist(data[:,cnt], bins=25, alpha=0.3)
        plt.xlabel(feature_dict[cnt])
        plt.legend(loc='upper right', fancybox=True, fontsize=8)

# Displays the Iris dataset in a tridimensional setting
def display_Iris(data,labels):
    fig=plt.figure(figsize=(8,6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)    

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    
    if not(len(labels)>1):
        ax.scatter(data[:, 3], data[:, 0], data[:, 2], c='black')
    else:
        ax.scatter(data[:, 3], data[:, 0], data[:, 2], c=labels.astype(np.float))
        for name, label in [('Setosa', 0),
                            ('Versicolour', 1),
                            ('Virginica', 2)]:                               
                            ax.text3D(data[labels == label, 3].mean(),
                                      data[labels == label, 0].mean() + 1.5,
                                      data[labels == label, 2].mean(), name,
                                      horizontalalignment='center',
                                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))

# Defines clusters based on the Guassian distribution
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
