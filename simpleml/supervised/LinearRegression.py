import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-3):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.theta = np.zeros(self.n_features + 1)
        self._costs = []

        i = 0
        while True:
            y_pred = self.predict(X)
            self.theta -= self.alpha * self._gradient(X, y, y_pred)
            cost = self._cost(y, y_pred)
            self._costs.append(cost)

            if i >= self.max_iter or cost <= self.tol:
                break

        return self

    def predict(self, X):
        return self._identity(X @ self.theta)

    def _identity(self, z):
        return z

    def _cost(self, y_true, y_pred):
        return (1 / (2 * self.n_samples)) * sum((y_pred - y_true) ** 2)

    def _gradient(self, X, y_true, y_pred):
        diff = (y_pred - y_true).reshape((-1, 1))
        return (1 / self.n_samples) * sum(diff * X)


if __name__ == '__main__':
    X, y = make_regression(n_samples=500, n_features=10, 
                           n_informative=10, n_targets=1) 

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LinearRegression()
    model.fit(X, y)
    
    plt.plot(model._costs)
    plt.title('Costs')
    plt.show()
