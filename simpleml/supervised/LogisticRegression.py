import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.001, tol=0.2, max_iter=10000, lambd=0, threshold=0.5):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.lambd = lambd
        self.threshold = threshold

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.theta = np.zeros(self.n_features + 1)
        self._costs = []

        i = 0
        while True:
            y_pred = self.activation(X)
            grad = self._gradient(X, y, y_pred)
            self.theta -= self.alpha * grad
            cost = self._cost(y, y_pred)
            if cost < self.tol or i > self.max_iter:
                return self
            self._costs.append(cost)
            i += 1

        return self

    def activation(self, X):
        return self._sigmoid(X @ self.theta)

    def predict(self, X):
        return np.where(self.activation(X) >= self.threshold, 1.0, 0.0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost(self, y_true, y_pred):
        left = -y_true * np.log(y_pred)
        right = -(1.0 - y_true) * np.log(1.0 - y_pred)
        left[np.isnan(left)] = -np.inf
        right[np.isnan(right)] = -np.inf
        cost = (1 / self.n_samples) * sum(left + right)
        cost += (self.lambd / (2 * self.n_samples)) * sum(self.theta[1:] ** 2)
        return cost

    def _gradient(self, X, y_true, y_pred):
        error = np.reshape((y_pred - y_true), (-1, 1))
        grad = (1 / self.n_samples) * sum(error * X)
        grad[1:] += (self.lambd / self.n_samples) * self.theta[1:]
        return grad


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2) 

    model = LogisticRegression()
    model.fit(X, y)
    
    plt.plot(model._costs)
    plt.title('Costs')
    plt.show()

    print('theta:', model.theta)

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    print('Accuracy:', sum(model.predict(X) == y.T) / y.shape[0])

