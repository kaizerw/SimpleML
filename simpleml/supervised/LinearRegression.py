import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, alpha=1e-3, max_iter=1e4, tol=1e-4, lambd=0, 
                 method='batch_gradient_descent', show_cost_plot=False):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        self.method = method # Method to be used to optimize cost function
        self.show_cost_plot = show_cost_plot # If show plot of cost function

    def fit(self, X, y):
        n = X.shape[1]
        self.w = np.zeros(n)
        self.b = 0

        if self.method == 'batch_gradient_descent':
            if self.show_cost_plot:
                plt.show()
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                axes = plt.gca()
                axes.set_xlim(0, self.max_iter)
                axes.set_ylim(0, self.__cost(np.concatenate((self.w, [self.b])), 
                                             X, y, self.lambd))
                line, = axes.plot([], [], 'r-')

            for t in range(self.max_iter):
                params = np.concatenate((self.w, [self.b]))

                d = self.__gradient(params, X, y, self.lambd)
                dw, db = d[:-1], d[-1]
                
                self.w -= self.alpha * dw
                self.b -= self.alpha * db

                cost = self.__cost(params, X, y, self.lambd)

                if self.show_cost_plot:
                    line.set_xdata(np.concatenate((line.get_xdata(), [t])))
                    line.set_ydata(np.concatenate((line.get_ydata(), [cost])))
                    plt.draw()
                    plt.pause(1e-50)

                if cost <= self.tol:
                    break

            if self.show_cost_plot:
                plt.close()
                    
        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd)
            params = np.concatenate((self.w, [self.b]))
            res = minimize(self.__cost, params, 
                           jac=self.__gradient, args=args, 
                           method=self.method, options=options)
            self.w, self.b = res.x[:-1], res.x[-1]

        return self

    def __activation(self, X, w, b):
        return X @ w + b

    def predict(self, X):
        return self.__activation(X, self.w, self.b)

    def __gradient(self, params, X, y, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        a = self.__activation(X, w, b)
        dz = a - y
        dw = (1 / m) * sum((dz * X.T).T)
        db = (1 / m) * sum(dz)
        dw += ((lambd / m) * sum(w))
        return np.concatenate((dw, [db]))

    def __cost(self, params, X, y, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        a = self.__activation(X, w, b)
        dz = a - y
        cost = (1 / (2 * m)) * sum(dz ** 2)
        cost += ((lambd / (2 * m)) * sum(w))
        return cost
