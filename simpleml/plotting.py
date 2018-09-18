import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_regression(X, y, model):
    plt.figure()
    plt.scatter(X, y, c='r')
    X_line = np.linspace(X.min(), X.max(), 1000)
    y_line = model.predict(np.reshape(X_line, (-1, 1)))
    plt.plot(X_line, y_line, c='b')
    plt.show()


def plot_decision_boundary(X, y, model):
    plt.figure()
    x_grid = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 1000)
    y_grid = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 1000)
    xx, yy = np.meshgrid(x_grid, y_grid)
    z = model.predict(np.hstack((np.reshape(xx, (-1, 1)), np.reshape(yy, (-1, 1)))))
    plt.contourf(xx, yy, z.reshape((1000, 1000)), cmap=cm.coolwarm, alpha=0.5)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='b')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='r')
    plt.show()


def plot_clustering(X, y_pred, model):
    centroids = model.centroids
    plt.figure()
    plt.scatter(X[y_pred==0, 0], X[y_pred==0, 1], c='r', alpha=0.5)
    plt.scatter(X[y_pred==1, 0], X[y_pred==1, 1], c='g', alpha=0.5)
    plt.scatter(X[y_pred==2, 0], X[y_pred==2, 1], c='b', alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=120, c='k')
    plt.show()
