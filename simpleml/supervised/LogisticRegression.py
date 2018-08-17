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
        X = np.hstack((np.ones((self.n_samples, 1)), X))
        y = y.reshape((-1, 1))

        n_classes = len(np.unique(y))

        if n_classes > 2:
            self.classifiers = n_classes

        y_true = np.zeros((y.shape[0], n_classes))
        for classe in range(n_classes):
            y_true[np.where(y==classe), classe] = 1

        if self.classifiers == 1:
            y_true = y
 
        self.theta = np.zeros((self.n_features + 1, self.classifiers))
        
        self.costs = []

        for classifier in range(self.classifiers):
            
            i = 0
            while True:
                theta = self.theta[:, classifier]
                y_pred = self.activation(X, theta)
                y = y_true[:, classifier].reshape((-1, 1))

                grad = self.gradient(X, y, y_pred, theta)
                self.theta[:, classifier] -= self.alpha * grad
                cost = self.cost(y, y_pred, theta)
                self.costs.append(cost)

                if i >= self.max_iter or cost <= self.tol:
                    break
                
                i += 1

        return self

    def activation(self, X, theta):
        return self.sigmoid(X @ theta).reshape((-1, 1))

    def predict(self, X):
        if self.classifiers > 2:
            y = np.zeros((X.shape[0], 1))
            for i in range(X.shape[0]):
                y[i, 0] = np.argmax([self.activation(X[i, :], self.theta[:, classifier]) 
                                     for classifier in range(self.classifiers)])
            return y
        return np.where(self.activation(X, self.theta[:, 0]) >= self.threshold, 1.0, 0.0)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def gradient(self, X, y_true, y_pred, theta):
        error = y_pred - y_true
        grad = (1 / self.n_samples) * sum(error * X)
        grad[1:] += (self.lambd / self.n_samples) * theta[1:]
        return grad

    def cost(self, y_true, y_pred, theta):
        left = -y_true * np.log(y_pred)
        right = -(1.0 - y_true) * np.log(1.0 - y_pred)
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
    
    plt.plot(model.costs)
    plt.title('Costs')
    plt.show()

    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y = y.reshape((-1, 1))
   
    print('Accuracy:', sum(model.predict(X) == y) / y.shape[0])
