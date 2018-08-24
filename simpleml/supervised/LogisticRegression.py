import numpy as np


class LogisticRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-4, lambd=0, threshold=0.5):
        self.alpha = alpha # Learning rate
        self.max_iter = max_iter # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        self.threshold = threshold # Threshold classification

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        self.n_samples, self.n_features = X.shape
        self.X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.y = y
 
        self.theta = np.zeros(self.n_features + 1)
        self.costs = []

        i = 0
        while True:
            y_pred = self._activation(self.X)
            y_true = y

            grad = self._gradient(y_true, y_pred)
            self.theta -= self.alpha * grad
            cost = self._cost(y_true, y_pred)
            self.costs.append(cost)

            if i >= self.max_iter or cost <= self.tol:
                break
            
            i += 1

        return self

    def _activation(self, X):
        return self._sigmoid(self.theta @ X.T)

    def predict(self, X):
        n_samples_test = X.shape[0]
        if not np.all(X[:, 0] == np.ones(n_samples_test)):
            X = np.hstack((np.ones((n_samples_test, 1)), X))
        return np.where(self._activation(X) >= self.threshold, 1, 0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _gradient(self, y_true, y_pred):
        error = y_pred - y_true
        grad = (1 / self.n_samples) * sum((error * self.X.T).T)
        grad[1:] += (self.lambd / self.n_samples) * self.theta[1:]
        return grad

    def _cost(self, y_true, y_pred):
        left = -y_true * np.log(y_pred)
        right = -(1 - y_true) * np.log(1 - y_pred)
        left[np.isnan(left)] = -np.inf
        right[np.isnan(right)] = -np.inf
        cost = (1 / self.n_samples) * sum(left + right)
        cost += (self.lambd / (2 * self.n_samples)) * sum(self.theta[1:] ** 2)
        return cost
