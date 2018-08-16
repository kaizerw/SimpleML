import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.001, n_iter=10000, lambd=0, threshold=0.5):
        self.alpha = alpha
        self.n_iter = n_iter
        self.lambd = lambd
        self.threshold = threshold
        self.classifiers = 1

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        y = np.reshape(y, (y.shape[0], 1))

        n_classes = len(np.unique(y))

        if n_classes > 2:
            self.classifiers = n_classes

        y_true = np.zeros((y.shape[0], n_classes))
        for classe in range(n_classes):
            y_true[np.where(y==classe), classe] = 1
 
        self.theta = np.zeros((self.n_features + 1, self.classifiers))
        
        self._costs = []

        for classifier in range(self.classifiers):
            for i in range(self.n_iter):
                theta = self.theta[:, classifier]
                y_pred = self.activation(X, theta)
                y = np.reshape(y_true[:, classifier], (-1, 1))

                grad = self._gradient(X, y, y_pred, theta)
                self.theta[:, classifier] -= self.alpha * grad
                cost = self._cost(y, y_pred, theta)
                self._costs.append(cost)

        return self

    def activation(self, X, theta):
        activation = self._sigmoid(X @ theta)
        return np.reshape(activation, (-1, 1))

    def predict(self, X):
        if self.classifiers > 2:
            return np.argmax([self.activation(X, self.theta[:, classifier]) 
                              for classifier in range(self.classifiers)])
        return np.where(self.activation(X) >= self.threshold, 1.0, 0.0)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost(self, y_true, y_pred, theta):
        left = -y_true * np.log(y_pred)
        right = -(1.0 - y_true) * np.log(1.0 - y_pred)
        left[np.isnan(left)] = -np.inf
        right[np.isnan(right)] = -np.inf
        cost = (1 / self.n_samples) * sum(left + right)
        cost += (self.lambd / (2 * self.n_samples)) * sum(theta[1:] ** 2)
        return cost

    def _gradient(self, X, y_true, y_pred, theta):
        error = y_pred - y_true
        grad = (1 / self.n_samples) * sum(error * X)
        grad[1:] += (self.lambd / self.n_samples) * theta[1:]
        return grad


if __name__ == '__main__':
    # TODO: Test deeply LogisticRegression with multiclasses

    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5) 

    model = LogisticRegression()
    model.fit(X, y)
    
    plt.plot(model._costs)
    plt.title('Costs')
    plt.show()

    print('theta:', model.theta)

    X = np.hstack((np.ones((X.shape[0], 1)), X))

    y = np.reshape(y, (y.shape[0], 1))
    print('Accuracy:', sum(model.predict(X) == y) / y.shape[0])
