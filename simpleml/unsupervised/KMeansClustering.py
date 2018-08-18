import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class KMeansClustering:

    def __init__(self, k=5, n_iter=1e3, p=2):
        self.k = k # Number of clusters
        self.n_iter = int(n_iter) # Number of iterations
        self.p = p # Power parameter to Minkowski distance

    # Minkowski distance
    def metric(self, x1, x2):
        return sum(abs(i - j) ** self.p for i, j in zip(x1, x2)) ** (1 / self.p)

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initializing K cluster centroids
        self.centroids = np.random.random((self.k, n_features))

        for _ in range(self.n_iter):
            # Cluster assignment step
            # C[i] = k: centroid k is the closest one to sample i
            C = {}
            for i in range(n_samples):
                C[i] = np.argmin([self.metric(X[i, :], self.centroids[j, :]) for j in range(self.k)])

            # Move centroid step
            for j in range(self.k):
                # Collecting ids of samples assigned to centroid j
                idx = [k for k, v in C.items() if v == j]
                if len(idx) > 0:
                    self.centroids[j, :] = np.mean(X[idx, :], axis=0)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = np.argmin([self.metric(X[i, :], self.centroids[j, :]) for j in range(self.k)])
        return y_pred


if __name__ == '__main__':
    X, y = make_blobs(n_samples=500, n_features=2, centers=5)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    model = KMeansClustering(k=5)
    model.fit(X)
    y = model.predict(X)

    plt.title('KMeans clustering with k=5')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='r', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='g', alpha=0.5)
    plt.scatter(X[y==2, 0], X[y==2, 1], c='b', alpha=0.5)
    plt.scatter(X[y==3, 0], X[y==3, 1], c='y', alpha=0.5)
    plt.scatter(X[y==4, 0], X[y==4, 1], c='m', alpha=0.5)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker='o', s=120, c='k')
    plt.show()
