import numpy as np
from scipy.optimize import minimize


class LogisticRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-4, lambd=0, 
                 threshold=0.5, method='batch__gradient_descent'):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        self.threshold = threshold # Threshold classification
        self.method = method # Method to be used to optimize cost function

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        n = X.shape[1]
        
        self.w = np.zeros(n)
        self.b = 0

        if self.method == 'batch__gradient_descent':
            for _ in range(self.max_iter):
                params = np.concatenate((self.w, [self.b]))

                d = self.__gradient(params, X, y, self.lambd)
                dw, db = d[:-1], d[-1]

                self.w -= self.alpha * dw
                self.b -= self.alpha * db

                cost = self.__cost(params, X, y, self.lambd)
                
                if cost <= self.tol:
                    break
                    
        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd)
            params = np.concatenate((self.w, [self.b]))
            res = minimize(self.__cost, params, 
                           jac=self.__gradient, args=args, 
                           method=self.method, options=options)
            self.w, self.b = res.x[:-1], res.x[-1]

        return self

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __activation(self, X, w, b):
        return self._sigmoid(X @ w + b)

    def predict(self, X):
        activation = self.__activation(X, self.w, self.b)
        return np.where(activation >= self.threshold, 1, 0)

    def predict_proba(self, X):
        activation = self.__activation(X, self.w, self.b)
        pos = np.reshape(activation, (-1, 1))
        neg = 1 - pos
        return np.hstack((neg, pos))

    def __gradient(self, params, X, y_true, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        z = self.__activation(X, w, b)
        dz = z - y_true
        dw = (1 / m) * sum((dz * X.T).T)
        db = (1 / m) * sum(dz)
        dw += (lambd / m) * w
        return np.concatenate((dw, [db]))

    def __cost(self, params, X, y_true, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        z = self.__activation(X, w, b)
        dz = -y_true * np.log(z) - (1 - y_true) * np.log(1 - z)
        cost = (1 / m) * sum(dz)
        cost += (lambd / (2 * m)) * sum(w ** 2)
        return cost
