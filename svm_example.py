import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 1. Generate sample data
X, y = make_blobs(n_samples=100, centers=2, random_state=6)

# 2. Train the SVM classifier
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# 3. Plot the decision boundary
plt.figure(figsize=(8, 6))

# Plot points
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# Draw decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[0], linestyles=['-'])
ax.contour(XX, YY, Z, colors='k', levels=[-1, 1], linestyles=['--'])

# Highlight support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=100, linewidth=1, facecolors='none', edgecolors='k')

plt.title("SVM - Linear Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
