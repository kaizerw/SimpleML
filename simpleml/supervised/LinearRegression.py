import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-3):
        self.alpha = alpha # Learning rate
        self.max_iter = max_iter # Max iterations
        self.tol = tol # Error tolerance

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.theta = np.zeros(self.n_features + 1)
        self.costs = []

        i = 0
        while True:
            y_pred = self.predict(X)
            self.theta -= self.alpha * self.gradient(X, y, y_pred)
            cost = self.cost(y, y_pred)
            self.costs.append(cost)

            if i >= self.max_iter or cost <= self.tol:
                break

        return self

    def predict(self, X):
        return self.identity(X @ self.theta)

    def identity(self, z):
        return z

    def gradient(self, X, y_true, y_pred):
        error = (y_pred - y_true).reshape((-1, 1))
        return (1 / self.n_samples) * sum(error * X)

    def cost(self, y_true, y_pred):
        error = y_pred - y_true
        return (1 / (2 * self.n_samples)) * sum(error ** 2)


if __name__ == '__main__':
    X, y = make_regression(n_samples=500, n_features=10, 
                           n_informative=10, n_targets=1) 

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LinearRegression()
    model.fit(X, y)
    
    plt.plot(model.costs)
    plt.title('Costs')
    plt.show()
