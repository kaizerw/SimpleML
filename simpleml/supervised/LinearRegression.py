import numpy as np
from scipy.optimize import minimize


class LinearRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-4, lambd=0, 
                 method='batch_gradient_descent'):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        self.method = method # Method to be used to optimize cost function

    def fit(self, X, y):
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

    def _activation(self, X, theta):
        return X @ theta

    def predict(self, X):
        n_samples_test = X.shape[0]
        if X.shape[1] != self.theta.shape[0]:
            X = np.hstack((np.ones((n_samples_test, 1)), X))
        return self._activation(X, self.theta)

    def _gradient(self, theta, X, y_true, lambd):
        n_samples = X.shape[0]
        y_pred = self._activation(X, theta)
        error = y_pred - y_true
        grad = (1 / n_samples) * sum((error * X.T).T)
        grad[1:] += ((lambd / n_samples) * sum(theta[1:]))
        return grad

    def _cost(self, theta, X, y_true, lambd):
        n_samples = X.shape[0]
        y_pred = self._activation(X, theta)
        error = y_pred - y_true
        cost = (1 / (2 * n_samples)) * sum(error ** 2)
        cost += ((lambd / (2 * n_samples)) * sum(theta[1:]))
        return cost
