import numpy as np
from scipy.optimize import minimize


class LogisticRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-4, lambd=0, 
                 threshold=0.5, method='batch_gradient_descent'):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        self.threshold = threshold # Threshold classification
        self.method = method # Method to be used to optimize cost function

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        n_samples, n_features = X.shape
        X = np.hstack((np.ones((n_samples, 1)), X))
        
        self.theta = np.zeros(n_features + 1)
        self.costs = []

        if self.method == 'batch_gradient_descent':
            for _ in range(self.max_iter):
                grad = self._gradient(self.theta, X, y, self.lambd)
                self.theta -= self.alpha * grad

                cost = self._cost(self.theta, X, y, self.lambd)
                self.costs.append(cost)

                if cost <= self.tol:
                    break
                    
        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd)
            res = minimize(self._cost, self.theta, 
                           jac=self._gradient, args=args, 
                           method=self.method, options=options)
            self.theta = res.x

        return self

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _activation(self, X, theta):
        n_samples_test = X.shape[0]
        if X.shape[1] != theta.shape[0]:
            X = np.hstack((np.ones((n_samples_test, 1)), X))
        return self._sigmoid(theta @ X.T)

    def predict(self, X):
        return np.where(self._activation(X, self.theta) >= self.threshold, 1, 0)

    def _gradient(self, theta, X, y_true, lambd):
        n_samples = X.shape[0]
        y_pred = self._activation(X, theta)
        error = y_pred - y_true
        grad = (1 / n_samples) * sum((error * X.T).T)
        grad[1:] += (lambd / n_samples) * theta[1:]
        return grad

    def _cost(self, theta, X, y_true, lambd):
        n_samples = X.shape[0]
        y_pred = self._activation(X, theta)
        left = -y_true * np.log(y_pred)
        right = -(1 - y_true) * np.log(1 - y_pred)
        left[np.isnan(left)] = -np.inf
        right[np.isnan(right)] = -np.inf
        cost = (1 / n_samples) * sum(left + right)
        cost += (lambd / (2 * n_samples)) * sum(theta[1:] ** 2)
        return cost
