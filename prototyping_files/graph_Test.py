import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
## initial prep -
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X4 = [[1,2,3],[3,4,5],[1,2,5],[1,3,5],[3,3,3],[1,1,1],[2,2,2],[3,3,3]]
Y4 = [[0],[1],[1],[0]]
X4 = np.array(X4)
Y4 = np.array(Y4)
X= np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[1,0,0,1,0],[1,0,0,0,1]])

y= np.array([0,1,1,1,0])
#y = y.reshape(1, -1)

#clf = svm.SVC(kernel='linear', C=1000)
clf = svm.SVC()
clf.fit(X, y)
#ax.scatter(X4[:, 0], X4[:, 1], X4[:, 2],s=30,cmap=plt.cm.Paired)
testPredict = np.array([1,0,0,1,1])
testPredict = testPredict.reshape(1, -1)
print(clf.predict(testPredict))
print(clf.predict([[1.,0.,0.,0.,0.]]))
print("Support Vec Info:")
print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)

# printing a 3d plot
#plt.show()