import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

X4 = [[1,2,3],
      [3,4,5],
      [1,2,5],
      [1,3,5]]


y4 = [[0],[1],[1],[0]]

#X, y = make_blobs(n_samples=40, centers=2, random_state=6)
X = [[6.37734541, -10.61510727],
     [6.50072722,  -3.82403586],
     [4.29225906,  -8.99220442],
 [7.39169472,  -3.1266933],
 [7.64306311, -10.02356892],
 [8.68185687,  -4.53683537],
 [5.37042238,  -2.44715237],
 [9.24223825,  -3.88003098],
 [5.73005848,  -4.19481136],
 [7.9683312,  -3.23125265],
 [7.37578372,  -8.7241701 ],
 [6.95292352,  -8.22624269],
 [8.21201164,  -1.54781358],
 [6.85086785,  -9.92422452],
 [5.64443032,  -8.21045789],
 [10.48848359,  -2.75858164],
 [7.27059007,  -4.84225716],
 [6.29784608, -10.53468031],
 [9.42169269,  -2.6476988 ],
 [8.98426675,  -4.87449712],
 [6.6008728,   -8.07144707],
 [5.95313618,  -6.82945967],
 [6.87151089, -10.18071547],
 [6.26221548,  -8.43925752],
 [7.97164446,  -3.38236058],
 [7.67619643,  -2.82620437],
 [7.92736799,  -9.7615272 ],
 [5.86311158, -10.19958738],
 [8.07502382,  -4.25949569],
 [6.78335342,  -8.09238614],
 [7.89359985,  -7.41655113],
 [6.04907774,  -8.76969991],
 [6.77811308,  -9.80940478],
 [8.71445065,  -2.41730491],
 [8.49142837,  -2.54974889],
 [9.49649411,  -3.7902975 ],
 [7.52132141,  -2.12266605],
 [6.3883927,  -9.25691447],
 [7.93333064,  -3.51553205],
 [6.86866543, -10.02289012]]

X = np.array(X)
#####
y = [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0,
 1, 0, 1]
# #  fitting the model here - may need to change so that it can handle 3 feature vectors
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(X, y)
# plot.scatter (produces scatter plot in matlab)
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
# plot the decision function
ax = plt.gca(projection='3d')
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title("Seperation of cyberbullying (red) vs Non-Cyberbullying (blue) cases")

plt.show()
'''
DATA: Uncomment when needed --- 

X = [[6.37734541, -10.61510727],
     [6.50072722,  -3.82403586],
     [4.29225906,  -8.99220442],
 [7.39169472,  -3.1266933],
 [7.64306311, -10.02356892],
 [8.68185687,  -4.53683537],
 [5.37042238,  -2.44715237],
 [9.24223825,  -3.88003098],
 [5.73005848,  -4.19481136],
 [7.9683312,  -3.23125265],
 [7.37578372,  -8.7241701 ],
 [6.95292352,  -8.22624269],
 [8.21201164,  -1.54781358],
 [6.85086785,  -9.92422452],
 [5.64443032,  -8.21045789],
 [10.48848359,  -2.75858164],
 [7.27059007,  -4.84225716],
 [6.29784608, -10.53468031],
 [9.42169269,  -2.6476988 ],
 [8.98426675,  -4.87449712],
 [6.6008728,   -8.07144707],
 [5.95313618,  -6.82945967],
 [6.87151089, -10.18071547],
 [6.26221548,  -8.43925752],
 [7.97164446,  -3.38236058],
 [7.67619643,  -2.82620437],
 [7.92736799,  -9.7615272 ],
 [5.86311158, -10.19958738],
 [8.07502382,  -4.25949569],
 [6.78335342,  -8.09238614],
 [7.89359985,  -7.41655113],
 [6.04907774,  -8.76969991],
 [6.77811308,  -9.80940478],
 [8.71445065,  -2.41730491],
 [8.49142837,  -2.54974889],
 [9.49649411,  -3.7902975 ],
 [7.52132141,  -2.12266605],
 [6.3883927,  -9.25691447],
 [7.93333064,  -3.51553205],
 [6.86866543, -10.02289012]]
 
X = [[6.37734541, -10.61510727,1],
 [6.50072722,  -3.82403586,1],
 [4.29225906,  -8.99220442,1],
 [7.39169472,  -3.1266933,1],
 [7.64306311, -10.02356892,1],
 [8.68185687,  -4.53683537,1],
 [5.37042238,  -2.44715237,1],
 [9.24223825,  -3.88003098,1],
 [5.73005848,  -4.19481136,1],
 [7.9683312,  -3.23125265,1],
 [7.37578372,  -8.7241701,1],
 [6.95292352,  -8.22624269,1],
 [8.21201164,  -1.54781358,1],
 [6.85086785,  -9.92422452,1],
 [5.64443032,  -8.21045789,1],
 [10.48848359,  -2.75858164,1],
 [7.27059007,  -4.84225716,1],
 [6.29784608, -10.53468031,1],
 [9.42169269,  -2.6476988,1],
 [8.98426675,  -4.87449712,1],
 [6.6008728,   -8.07144707,1],
 [5.95313618,  -6.82945967,1],
 [6.87151089, -10.18071547,1],
 [6.26221548,  -8.43925752,1],
 [7.97164446,  -3.38236058,1],
 [7.67619643,  -2.82620437,1],
 [7.92736799,  -9.7615272,1],
 [5.86311158, -10.19958738,1],
 [8.07502382,  -4.25949569,1],
 [6.78335342,  -8.09238614,1],
 [7.89359985,  -7.41655113,1],
 [6.04907774,  -8.76969991,1],
 [6.77811308,  -9.80940478,1],
 [8.71445065,  -2.41730491,1],
 [8.49142837,  -2.54974889,1],
 [9.49649411,  -3.7902975,1],
 [7.52132141,  -2.12266605,1],
 [6.3883927,  -9.25691447,1],
 [7.93333064,  -3.51553205,1],
 [6.86866543, -10.02289012,1]]
 '''