import numpy as np


class KMeansClustering:

    def __init__(self, k=5, max_iter=1e3, tol=1e-4, p=2, repetitions=10):
        self.k = k # Number of clusters
        self.max_iter = int(max_iter) # Number of iterations
        self.tol = tol # Stop criteria: min cost change per iteration
        self.p = p # Power parameter to Minkowski distance
        self.repetitions = repetitions # Number of algorithm repetitions

    # Minkowski distance
    def _metric(self, x1, x2):
        return sum(abs(i - j) ** self.p for i, j in zip(x1, x2)) ** (1 / self.p)

    def fit(self, X):
        self.n_samples, self.n_features = X.shape

        self.centroids, min_cost = None, np.inf
        for _ in range(self.repetitions):
            # Initializing k cluster centroids
            centroids = np.random.random((self.k, self.n_features))

            previous_cost = np.inf
            for _ in range(self.max_iter):
                # Cluster assignment step
                # C[i] = k: centroid k is the closest one to sample i
                C = {}
                for i in range(self.n_samples):
                    C[i] = np.argmin([self._metric(X[i, :], centroids[j, :]) 
                                    for j in range(self.k)])

                # Move centroid step
                for j in range(self.k):
                    # Collecting ids of samples assigned to centroid j
                    idx = [k for k, v in C.items() if v == j]
                    if len(idx) > 0:
                        centroids[j, :] = np.mean(X[idx, :], axis=0)

                current_cost = self._cost(X, centroids, C)
                if abs(previous_cost - current_cost) < self.tol:
                    break
                previous_cost = current_cost

            if current_cost < min_cost:
                self.centroids = centroids
                min_cost = current_cost
        
        return self

    def _cost(self, X, centroids, C):
        return sum([self._metric(X[i, :], centroids[C[i], :]) 
                   for i in range(self.n_samples)])

    def predict(self, X):
        n_samples_test = X.shape[0]
        y_pred = np.zeros(n_samples_test)
        # Predict class from the closest cluster centroid
        for i in range(n_samples_test):
            y_pred[i] = np.argmin([self._metric(X[i, :], self.centroids[j, :]) 
                                  for j in range(self.k)])
        return y_pred
