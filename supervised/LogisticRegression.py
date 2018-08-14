import numpy as np

class LogisticRegression:

    def __init__(self, alpha=0.001, tol=0.2, lambd=0, threshold=0.5):
        self.alpha = alpha
        self.tol = tol
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
            if cost < self.tol:
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
        diff = np.reshape((y_pred - y_true), (-1, 1))
        grad = (1 / self.n_samples) * sum(diff * X)
        grad[1:] += (self.lambd / self.n_samples) * self.theta[1:]
        return grad


if __name__ == '__main__':
    data = np.loadtxt('logisticTest1.txt', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    model = LogisticRegression(lambd=1)
    model.fit(X, y)
    print('Costs:', model._costs)

    print('theta:', model.theta)

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    print('Accuracy:', sum(model.predict(X) == y.T) / y.shape[0])

    print('Predict:', model._sigmoid(np.array([1, 45, 85]) @ model.theta))
