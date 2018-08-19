import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-3, lambd=0, threshold=0.5):
        self.alpha = alpha # Learning rate
        self.max_iter = max_iter # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        self.threshold = threshold # Threshold classification
        self.classifiers = 1 # Number of classifiers

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = np.hstack((np.ones((self.n_samples, 1)), X))
        self.y = y

        n_classes = len(np.unique(self.y))
        if n_classes > 2:
            self.classifiers = n_classes

        y_true = np.zeros((self.n_samples, n_classes))
        for classe in range(n_classes):
            y_true[self.y == classe, classe] = 1
        if self.classifiers == 1:
            y_true[self.y == 1, 0] = 1
 
        self.theta = np.zeros((self.classifiers, self.n_features + 1))
        self.costs = []

        for classifier in range(self.classifiers):
            i = 0

            while True:
                theta = self.theta[classifier, :]
                y_pred = self._activation(self.X, theta)
                y = y_true[:, classifier]

                grad = self._gradient(y, y_pred, theta)
                theta -= self.alpha * grad
                cost = self._cost(y, y_pred, theta)
                self.costs.append(cost)

                if i >= self.max_iter or cost <= self.tol:
                    break
                
                i += 1

        return self

    def _activation(self, X, theta):
        return self._sigmoid(theta @ X.T)

    def predict(self, X):
        n_samples_test = X.shape[0]
        if self.classifiers > 2:
            y = np.zeros(n_samples_test)
            for i in range(n_samples_test):
                y[i] = np.argmax([self._activation(X[i, :], self.theta[classifier, :]) 
                                  for classifier in range(self.classifiers)])
            return y
        return np.where(self._activation(X, self.theta[0, :]) >= self.threshold, 1, 0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _gradient(self, y_true, y_pred, theta):
        error = y_pred - y_true
        grad = (1 / self.n_samples) * sum((error * self.X.T).T)
        grad[1:] += (self.lambd / self.n_samples) * theta[1:]
        return grad

    def _cost(self, y_true, y_pred, theta):
        left = -y_true * np.log(y_pred)
        right = -(1 - y_true) * np.log(1 - y_pred)
        left[np.isnan(left)] = -np.inf
        right[np.isnan(right)] = -np.inf
        cost = (1 / self.n_samples) * sum(left + right)
        cost += (self.lambd / (2 * self.n_samples)) * sum(theta[1:] ** 2)
        return cost


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LogisticRegression()
    model.fit(X, y)
    
    # plt.plot(model.costs)
    # plt.title('Costs')
    # plt.show()

    n_samples_test = X.shape[0]
    X = np.hstack((np.ones((n_samples_test, 1)), X))
    print('Accuracy:', sum(model.predict(X) == y) / n_samples_test)
