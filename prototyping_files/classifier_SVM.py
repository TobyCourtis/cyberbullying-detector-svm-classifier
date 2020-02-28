"""
=========================================
SVM: Maximum margin separating hyperplane
=========================================

Plot the maximum margin separating hyperplane within a two-class
separable dataset using a Support Vector Machine classifier with
linear kernel.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs


# we create 40 separable points
# Here X = the co-ordinates of each point ( believed to be the vector )
# Here y = a vector with classification of the point in X (Is it YES cyberbullying (1) or NO cyberbullying (0)
X, y = make_blobs(n_samples=40, centers=1, random_state=6)

y[::2] = 1
print(X)
print("---------")
print(y)

# fit the model, don't regularize for illustration purposes
clf = svm.SVC(kernel='linear',C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
levels = [-1, 0, 1]
ax.contour(XX, YY, Z, colors='k', levels=levels, alpha=0.5, linestyles=['-.', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("Seperation of cyberbullying (red) vs Non-Cyberbullying (blue) cases")



plt.show()