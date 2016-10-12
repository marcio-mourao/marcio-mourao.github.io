print(__doc__)

import aux_funcs
import numpy as np

from sklearn import datasets
import matplotlib.pyplot as plt

# Load the iris dataset
iris = datasets.load_iris()

# Obtains the raw data
X = iris.data

# Some exploration of the data
aux_funcs.display_Iris(data=X,labels=[])
print('\nMean for each attribute: \n', np.mean(X,axis=0))
aux_funcs.display_Iris_Dist(data=X,labels=[])

### Apply PCA methodology (Hard solution)

from sklearn.preprocessing import StandardScaler

# Scale the data
X_std = StandardScaler().fit_transform(X)
print('\nMean for each attribute: \n', np.mean(X_std,axis=0))
# Obtain the covariance matrix
cov_mat = np.cov(X_std.T)
print('\nNumPy covariance matrix: \n%s' %cov_mat)

# Check scatter of some combinations
plt.figure()
plt.scatter(X_std[:,0],X_std[:,3])

# Obtain eigenvalues and eigenvectors of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' %eig_vals)
print('\nEigenvectors \n%s' %eig_vecs)

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)
print('\nEigenvalues in descending order:\n')
for i in eig_pairs: print(i[0])

# Obtain variance 
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('\nCumulative of the proportion of variance:\n', cum_var_exp)

# Obtain projection matrix using the two first components (>95%)
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))
print('\nProjection Matrix W:\n', matrix_w)

# Project onto the new reduced space
Y = X_std.dot(matrix_w)
# Plot outcome
aux_funcs.display_Iris_PCA(data=Y,labels=[])

# Check covariance matrix of the data in the reduced dimensional space
Y_std = StandardScaler().fit_transform(Y)
cov_mat2 = np.cov(Y_std.T)
print('\nCovariance matrix in the reduced space: \n',cov_mat2)

### Apply PCA methodology (Simple solution)

from sklearn.decomposition import PCA

# Creates the main PCA object
pca = PCA(n_components=0.95)
# Fits the data
pca_fit = pca.fit(X)
# Obtain the "scores" - projection onto the new reduced space
Y = pca_fit.transform(X)
# Plot outcome
aux_funcs.display_Iris_PCA(data=Y,labels=[])

### Some exploratory visualization with classes that we actually know

# Obtains CLass label IDs
y = iris.target
aux_funcs.display_Iris(data=X,labels=y)
aux_funcs.display_Iris_Dist(data=X,labels=y)
aux_funcs.display_Iris_PCA(data=Y,labels=y)
