"""
====================================================
FROVOCO Classification
====================================================

Sample usage of FROVOCO classification.
It will plot the decision boundaries for each class.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

from frlearn.base import discretise_scores
from frlearn.ensembles import FROVOCO

# import example data and reduce to 2 dimensions
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
X = np.concatenate([X[:5], X[50:67], X[100:]], axis=0)
y = np.concatenate([y[:5], y[50:67], y[100:]], axis=0)

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

# create an instance of the FROVOCO Classifier and construct the model.
clf = FROVOCO()
model = clf.construct(X, y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model.query(np.c_[xx.ravel(), yy.ravel()])
Z = discretise_scores(Z, labels=model.classes)

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
            edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('3-Class classification with FROVOCO')

plt.show()

