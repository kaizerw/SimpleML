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
        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0

        if self.method == 'batch_gradient_descent':
            for _ in range(self.max_iter):
                params = np.concatenate((self.w, [self.b]))

                d = self._gradient(params, X, y, self.lambd)
                dw, db = d[:-1], d[-1]
                self.w -= self.alpha * dw
                self.b -= self.alpha * db

                cost = self._cost(params, X, y, self.lambd)

                if cost <= self.tol:
                    break
                    
        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd)
            params = np.concatenate((self.w, [self.b]))
            res = minimize(self._cost, params, 
                           jac=self._gradient, args=args, 
                           method=self.method, options=options)
            self.w, self.b = res.x[:-1], res.x[-1]

        return self

    def _activation(self, X, w, b):
        return (X @ w) + b

    def predict(self, X):
        return self._activation(X, self.w, self.b)

    def _gradient(self, params, X, y_true, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        y_pred = self._activation(X, w, b)
        error = y_pred - y_true
        dw = (1 / m) * sum((error * X.T).T)
        db = (1 / m) * sum(error)
        dw += ((lambd / m) * sum(w))
        return np.concatenate((dw, [db]))

    def _cost(self, params, X, y_true, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        y_pred = self._activation(X, w, b)
        error = y_pred - y_true
        cost = (1 / (2 * m)) * sum(error ** 2)
        cost += ((lambd / (2 * m)) * sum(w))
        return cost
