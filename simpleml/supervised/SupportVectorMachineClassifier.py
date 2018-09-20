import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial


class SupportVectorMachineClassifier:

    def __init__(self, max_iter=1e4, tol=1e-4, C=0):
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.C = C # Regularization constant
        
    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        n, m = X.shape
        y = np.where(y==1, 1, -1)
        
        self.w, self.b = np.zeros(m), 0
        self.zeta = np.zeros(n)

        args = (X, y, self.C)
        params = np.concatenate((self.w, [self.b], self.zeta))
        res = minimize(self.__cost, params, args=args, 
                       constraints=[{'type': 'ineq', 
                                     'fun': partial(self.__constraint, i), 
                                     'args': args} for i in range(n)], 
                       bounds=[self.__bound(i, n, m) for i in range(len(params))])
        self.w, self.b, self.zera = res.x[:m], res.x[m:m + 1], res.x[m + 1:]

        print(self.w, self.b, self.zeta)

        return self

    def __activation(self, X, w, b):
        return X @ w + b

    def predict(self, X):
        activation = self.__activation(X, self.w, self.b)
        return np.where(activation >= 0, 1, 0)

    def predict_proba(self, X):
        activation = self.__activation(X, self.w, self.b)
        pos = np.reshape(activation, (-1, 1))
        neg = 1 - pos
        return np.hstack((neg, pos))

    def __cost(self, params, X, y, C):
        _, m = X.shape
        w, _, zeta = params[:m], params[m:m + 1], params[m + 1:]
        return sum((1 / 2) * (w ** 2) + C * sum(zeta))

    def __constraint(self, i, params, X, y, C):
        _, m = X.shape
        w, b, zeta = params[:m], params[m:m + 1], params[m + 1:]
        return y[i] * (X[i, :] @ w + b) - (1 - zeta[i])

    def __bound(self, i, n, m):
        if i >= m + 1:
            return (0, None)
        return (None, None)
