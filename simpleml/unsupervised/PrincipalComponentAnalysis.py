import numpy as np


class PrincipalComponentAnalysis:

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.sigma = (1 / self.n_samples) * (X.T @ X)
        self.U, self.S, self.V = np.linalg.svd(self.sigma)
        self.U = self.U[:, :self.n_components]

    def transform(self, X):
        return X @ self.U

    def inverse_transform(self, Z):
        return Z @ self.U.T
