import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


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


if __name__ == '__main__': 
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    pca = PrincipalComponentAnalysis(2)
    pca.fit(X)
    Z = pca.transform(X)
    print('Principal Components:', Z)

    X_approx = pca.inverse_transform(Z)
    print('Error X_approx:', sum(sum(X_approx - X)))
