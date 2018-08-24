import numpy as np


class LinearRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-3, lambd=0):
        self.alpha = alpha # Learning rate
        self.max_iter = max_iter # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.y = y
        self.theta = np.zeros(self.n_features + 1)
        self.costs = []

        i = 0
        while True:
            y_pred = self.predict(self.X)
            self.theta -= self.alpha * self._gradient(y_pred)
            cost = self._cost(y_pred)
            self.costs.append(cost)

            if i >= self.max_iter or cost <= self.tol:
                break
            
            i += 1

        return self

    def predict(self, X):
        return X @ self.theta

    def R2(self):
        y_pred = self.predict(self.X)
        y_mean = np.mean(self.y)

        ss_tot = sum((self.y - y_mean) ** 2)
        ss_res = sum((self.y - y_pred) ** 2)

        return 1 - (ss_res / ss_tot)

    def _gradient(self, y_pred):
        error = y_pred - self.y
        grad = (1 / self.n_samples) * sum((error * self.X.T).T)
        grad[1:] += ((self.lambd / self.n_samples) * sum(self.theta[1:]))
        return grad

    def _cost(self, y_pred):
        error = y_pred - self.y
        cost = (1 / (2 * self.n_samples)) * sum(error ** 2)
        cost += ((self.lambd / (2 * self.n_samples)) * sum(self.theta[1:]))
        return cost
