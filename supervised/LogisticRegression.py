import numpy as np

class LogisticRegression:

    def __init__(self, alpha=0.1, n_iter=1000, lambd=0, threshold=0.5):
        self.alpha = alpha
        self.n_iter = n_iter
        self.lambd = lambd
        self.threshold = threshold
        self._epsilon = 1e-10

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
        return self._sigmoid(X @ self.theta) >= self.threshold

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost(self, y_true, y_pred):
        cost = (1 / self.n_samples) * sum(-y_true * np.log(y_pred + self._epsilon) - (1 - y_true) * np.log(1 - y_pred + self._epsilon))
        cost += (self.lambd / (2 * self.n_samples)) * sum(self.theta[1:] ** 2)
        return cost

    def _gradient(self, X, y_true, y_pred):
        diff = np.reshape((y_pred - y_true), (-1, 1))
        grad = (1 / self.n_samples) * sum(diff * X)
        grad[1:] += (self.lambd / self.n_samples) * self.theta[1:]
        return grad


if __name__ == '__main__':
    # TODO: Test deeply Logistic Regression
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
