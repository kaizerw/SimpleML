import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class SupportVectorMachineClassifier:

    def __init__(self, max_iter=1e4, tol=1e-4, lambd=0):
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        
    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        _, m = X.shape
        y = np.where(y==1, 1, -1)
        
        self.w, self.b = np.zeros(m), 0

        args = (X, y, self.lambd)
        params = np.concatenate((self.w, [self.b]))
        res = minimize(self.__cost, params, args=args)
        self.w, self.b = res.x[:-1], res.x[-1]

        return self

    def __activation(self, X, w, b):
        return X @ w + b

    def predict(self, X):
        activation = self.__activation(X, self.w, self.b)
        return np.where(activation >= 0, 1, 0)

    def predict_proba(self, X):
        activation = self.__activation(X, self.w, self.b)
        activation = np.where(activation >= 0, 1, 0)
        pos = np.reshape(activation, (-1, 1))
        neg = 1 - pos
        return np.hstack((neg, pos))

    def __cost(self, params, X, y, lambd):
        n, _ = X.shape
        w, b = params[:-1], params[-1]
        zeta = np.max(np.vstack((np.zeros_like(y), 1 - y * (X @ w + b))), axis=0)
        cost = (1 / n) * sum(zeta)
        cost += lambd * np.linalg.norm(w)
        return cost
