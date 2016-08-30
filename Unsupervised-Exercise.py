import aux_funcs
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#KMeans Exercise
nPoints=800
nClasses=10
s2=0.1
[X,target]=aux_funcs.init_board_gauss(nPoints,nClasses,s2)
plt.figure(3,figsize=(10,10))
plt.scatter(X[:,0],X[:,1],c=target,s=nPoints*[100])
model=KMeans(n_clusters=5)
model.fit(X)
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c="k",s=nClasses*[300])
plt.show()


