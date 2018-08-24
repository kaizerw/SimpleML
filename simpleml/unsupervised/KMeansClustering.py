import numpy as np


class KMeansClustering:

    def __init__(self, k=5, n_iter=1e3, p=2):
        self.k = k # Number of clusters
        self.n_iter = int(n_iter) # Number of iterations
        self.p = p # Power parameter to Minkowski distance

    # Minkowski distance
    def _metric(self, x1, x2):
        return sum(abs(i - j) ** self.p for i, j in zip(x1, x2)) ** (1 / self.p)

    def fit(self, X):
        self.n_samples, self.n_features = X.shape

        # Initializing k cluster centroids
        self.centroids = np.random.random((self.k, self.n_features))

        for _ in range(self.n_iter):
            # Cluster assignment step
            # C[i] = k: centroid k is the closest one to sample i
            C = {}
            for i in range(self.n_samples):
                C[i] = np.argmin([self._metric(X[i, :], self.centroids[j, :]) for j in range(self.k)])

            # Move centroid step
            for j in range(self.k):
                # Collecting ids of samples assigned to centroid j
                idx = [k for k, v in C.items() if v == j]
                if len(idx) > 0:
                    self.centroids[j, :] = np.mean(X[idx, :], axis=0)

    def predict(self, X):
        n_samples_test = X.shape[0]
        y_pred = np.zeros(n_samples_test)
        # Predict class from the closest cluster centroid
        for i in range(n_samples_test):
            y_pred[i] = np.argmin([self._metric(X[i, :], self.centroids[j, :]) for j in range(self.k)])
        return y_pred
