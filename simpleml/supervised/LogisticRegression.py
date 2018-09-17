import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=1e-4, max_iter=1e4, tol=1e-4, lambd=0, 
                 beta1=0.9, beta2=0.999, threshold=0.5, 
                 method='batch_gradient_descent', show_cost_plot=False):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.lambd = lambd # Regularization constant
        self.beta1 = beta1 # Momentum constant
        self.beta2 = beta2 # RMSprop constant
        self.threshold = threshold # Threshold classification
        self.method = method # Method to be used to optimize cost function
        self.show_cost_plot = show_cost_plot # If show plot of cost function

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        n = X.shape[1]
        
        # He initialization
        self.w = np.random.randn(n) * np.sqrt(2 / n) 
        self.b = 0

        if self.method == 'batch_gradient_descent':
            VdW = np.zeros(self.w.shape)
            Vdb = 0

            SdW = np.zeros(self.w.shape)
            Sdb = 0

            for t in range(1, self.max_iter + 1):
                if self.show_cost_plot:
                    plt.show()
                    plt.xlabel('Iteration')
                    plt.ylabel('Cost')
                    axes = plt.gca()
                    axes.set_xlim(0, self.max_iter)
                    axes.set_ylim(0, self.__cost(np.concatenate((self.w, [self.b])), 
                                                X, y, self.lambd))
                    line, = axes.plot([], [], 'r-')

                params = np.concatenate((self.w, [self.b]))

                d = self.__gradient(params, X, y, self.lambd)
                dw, db = d[:-1], d[-1]

                # Momentum
                VdW = self.beta1 * VdW + (1 - self.beta1) * dw
                Vdb = self.beta1 * Vdb + (1 - self.beta1) * db

                VdW_c = VdW / (1 - self.beta1 ** t)
                Vdb_c = Vdb / (1 - self.beta1 ** t)

                # RMSprop
                SdW = self.beta2 * SdW + (1 - self.beta2) * dw ** 2
                Sdb = self.beta2 * Sdb + (1 - self.beta2) * db ** 2

                SdW_c = SdW / (1 - self.beta2 ** t)
                Sdb_c = Sdb / (1 - self.beta2 ** t)

                self.w -= self.alpha * (VdW_c / (np.sqrt(SdW_c) + 1e-8))
                self.b -= self.alpha * (Vdb_c / (np.sqrt(Sdb_c) + 1e-8))

                cost = self.__cost(params, X, y, self.lambd)

                if self.show_cost_plot:
                    line.set_xdata(np.concatenate((line.get_xdata(), [t])))
                    line.set_ydata(np.concatenate((line.get_ydata(), [cost])))
                    plt.draw()
                    plt.pause(1e-10)
                
                if cost <= self.tol:
                    break
                    
            if self.show_cost_plot:
                plt.close()
                    
        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd)
            params = np.concatenate((self.w, [self.b]))
            res = minimize(self.__cost, params, jac=self.__gradient, args=args, 
                           method=self.method, options=options)
            self.w, self.b = res.x[:-1], res.x[-1]

        return self

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __activation(self, X, w, b):
        return self.__sigmoid(X @ w + b)

    def predict(self, X):
        activation = self.__activation(X, self.w, self.b)
        return np.where(activation >= self.threshold, 1, 0)

    def predict_proba(self, X):
        activation = self.__activation(X, self.w, self.b)
        pos = np.reshape(activation, (-1, 1))
        neg = 1 - pos
        return np.hstack((neg, pos))

    def __gradient(self, params, X, y, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        a = self.__activation(X, w, b)
        dz = a - y
        dw = (1 / m) * sum((dz * X.T).T)
        db = (1 / m) * sum(dz)
        dw += (lambd / m) * w
        return np.concatenate((dw, [db]))

    def __cost(self, params, X, y, lambd):
        w, b = params[:-1], params[-1]
        m = X.shape[0]
        a = self.__activation(X, w, b)
        dz = -y * np.log(a) - (1 - y) * np.log(1 - a)
        cost = (1 / m) * sum(dz)
        cost += (lambd / (2 * m)) * sum(w ** 2)
        return cost
