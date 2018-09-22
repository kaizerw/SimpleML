import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from functools import partial


class SupportVectorMachineRegressor:

    def __init__(self, max_iter=1e4, tol=1e-4, epsilon=0.1):
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.epsilon = epsilon # SVR model coefficient
        
    def fit(self, X, y):
        n, m = X.shape
        
        self.w, self.b = np.zeros(m), 0

        args = (X, y, self.epsilon)
        params = np.concatenate((self.w, [self.b]))

        constraints = []
        constraints.extend([{'type': 'ineq', 'fun': partial(self.__constraints1, i), 'args': args} for i in range(n)])
        constraints.extend([{'type': 'ineq', 'fun': partial(self.__constraints2, i), 'args': args} for i in range(n)])

        res = minimize(self.__cost, params, args=args, 
                       constraints=constraints)
        self.w, self.b = res.x[:-1], res.x[-1]

        return self

    def __activation(self, X, w, b):
        return X @ w + b

    def predict(self, X):
        return self.__activation(X, self.w, self.b)

    def __cost(self, params, X, y, epsilon):
        w, _ = params[:-1], params[-1]
        cost = (1 / 2) * np.linalg.norm(w)
        return cost

    def __constraints1(self, i, params, X, y, epsilon):
        w, b = params[:-1], params[-1]
        return (y[i] + (X[i, :] @ w) + b) + epsilon

    def __constraints2(self, i, params, X, y, epsilon):
        w, b = params[:-1], params[-1]
        return ((X[i, :] @ w) - b + y[i]) + epsilon
