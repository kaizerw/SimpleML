import numpy as np


class LinearRegression:

    def __init__(self, alpha=0.1, n_iter=1000):
        self.alpha = alpha
        self.n_iter = n_iter

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.theta = np.random.randn(self.n_features + 1)
        self._costs = np.zeros(self.n_iter)

        for i in range(self.n_iter):
            y_pred = self.predict(X)
            self.theta -= self.alpha * self._gradient(X, y, y_pred)
            self._costs[i] = self._cost(y, y_pred)

        return self

    def predict(self, X):
        return self._identity(X @ self.theta)

    def _identity(self, z):
        return z

    def _cost(self, y_true, y_pred):
        return (1 / (2 * self.n_samples)) * sum((y_pred - y_true) ** 2)

    def _gradient(self, X, y_true, y_pred):
        diff = np.reshape((y_pred - y_true), (-1, 1))
        return (1 / self.n_samples) * sum(diff * X)


if __name__ == '__main__':
    data = np.loadtxt('housing.csv', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LinearRegression(n_iter=50000)
    model.fit(X, y)
    print('Costs:', model._costs)

    print('theta:', model.theta)

    x_test = np.array([1650, 3])
    x_test = (x_test - mu) / sigma
    x_test = np.hstack(([1], x_test))
    print('Predict:', model.predict(x_test))
