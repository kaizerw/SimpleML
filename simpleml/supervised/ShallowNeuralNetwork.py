import numpy as np
from scipy.optimize import minimize


class ShallowNeuralNetwork:

    def __init__(self, alpha=1e-5, max_iter=1e4, tol=1e-3, n_hid=25, lambd=0, 
                 activation='relu', method='batch_gradient_descent'):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.n_hid = n_hid # Number of neurons in the hidden layer
        self.lambd = lambd # Regularization constant
        self.method = method # Method to be used to optimize cost function
        if activation == 'sigmoid':
            self.activation = self.__sigmoid
            self.gradient = self.__sigmoid_gradient
        elif activation == 'tanh':
            self.activation = self.__tanh
            self.gradient = self.__tanh_gradient
        elif activation == 'relu':
            self.activation = self.__relu
            self.gradient = self.__relu_gradient

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        y = y.reshape((-1, 1))

        n_in = X.shape[1]
        n_out = 1
        
        self.W1 = np.random.randn(n_in, self.n_hid) * 1e-2
        self.b1 = np.zeros(self.n_hid)
        self.W2 = np.random.randn(self.n_hid, n_out) * 1e-2
        self.b2 = np.zeros(n_out)

        if self.method == 'batch_gradient_descent':
            for _ in range(self.max_iter):
                params = self.__zip_params(self.W1, self.b1, self.W2, self.b2)
                
                d = self.__gradient(params, X, y, self.lambd, 
                                    n_in, self.n_hid, n_out)
                dW1, db1, dW2, db2 = self.__unzip_params(d, n_in, 
                                                         self.n_hid, n_out)
                
                self.W1 -= self.alpha * dW1
                self.b1 -= self.alpha * db1
                self.W2 -= self.alpha * dW2
                self.b2 -= self.alpha * db2
                
                cost = self.__cost(params, X, y, self.lambd, n_in, 
                                   self.n_hid, n_out)

                if cost <= self.tol:
                    break

        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd, n_in, self.n_hid, n_out)
            params = self.__zip_params(self.W1, self.b1, self.W2, self.b2)
            res = minimize(self.__cost, params, jac=self.__gradient, args=args, 
                           method=self.method, options=options)
            params = res.x
            W1, b1, W2, b2 = self.__unzip_params(params, n_in, 
                                                 self.n_hid, n_out)
            self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2

        return self

    def __zip_params(self, W1, b1, W2, b2):
        return np.concatenate((W1.reshape(-1), b1.reshape(-1), 
                               W2.reshape(-1), b2.reshape(-1)))
    
    def __unzip_params(self, params, n_in, n_hid, n_out):
        begin_W1 = 0
        end_W1 = n_in * n_hid
        end_b1 = end_W1 + n_hid
        end_W2 = end_b1 + (n_hid * n_out)

        W1 = params[begin_W1:end_W1].reshape((n_in, n_hid))
        b1 = params[end_W1:end_b1]
        W2 = params[end_b1:end_W2].reshape((n_hid, n_out))
        b2 = params[end_W2:]
        return W1, b1, W2, b2

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def __sigmoid_gradient(self, z):
        return self.__sigmoid(z) * (1 - self.__sigmoid(z))

    def __tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    def __tanh_gradient(self, z):
        return 1 - self.__tanh(z) ** 2

    def __relu(self, z):
        return np.where(z < 0, 0, z)

    def __relu_gradient(self, z):
        return np.where(z < 0, 0, 1)

    def predict(self, X):   
        Z1, A1, Z2, A2 = self.__forward_pass(X, self.W1, self.b1, 
                                           self.W2, self.b2)
        return np.where(A2 > 0.5, 1, 0).reshape(-1)

    def predict_proba(self, X):
        Z1, A1, Z2, A2 = self.__forward_pass(X, self.W1, self.b1, 
                                           self.W2, self.b2)
        return A2.reshape(-1)

    def __forward_pass(self, X, W1, b1, W2, b2):
        Z1 = X @ W1 + b1
        A1 = self.activation(Z1)
        Z2 = A1 @ W2 + b2
        A2 = self.__sigmoid(Z2)
        return Z1, A1, Z2, A2

    def __backward_pass(self, Z1, A1, W1, Z2, A2, W2, X, y):
        m = X.shape[0]

        dZ2 = A2 - y
        dW2 = (1 / m) * A1.T @ dZ2
        db2 = (1 / m) * np.sum(dZ2, axis=0)

        dZ1 = dZ2 @ W2.T * self.gradient(Z1)
        dW1 = (1 / m) * X.T @ dZ1
        db1 = (1 / m) * np.sum(dZ1, axis=0)
        return dW1, db1, dW2, db2

    def __gradient(self, params, X, y, lambd, n_in, n_hid, n_out):
        W1, b1, W2, b2 = self.__unzip_params(params, n_in, n_hid, n_out)
        m = X.shape[0]

        # Forward pass
        Z1, A1, Z2, A2 = self.__forward_pass(X, W1, b1, W2, b2)

        # Backward pass
        dW1, db1, dW2, db2 = self.__backward_pass(Z1, A1, W1, Z2, A2, W2, X, y)        

        # Regularization
        dW1[:, 1:] += (lambd / m) * W1[:, 1:]
        dW2[:, 1:] += (lambd / m) * W2[:, 1:]

        return self.__zip_params(dW1, db1, dW2, db2)

    def __cost(self, params, X, y, lambd, n_in, n_hid, n_out):
        W1, b1, W2, b2 = self.__unzip_params(params, n_in, n_hid, n_out)
        m = X.shape[0]

        Z1, A1, Z2, A2 = self.__forward_pass(X, W1, b1, W2, b2)
        cost = -(1 / m) * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))
        cost += ((lambd / (2 * m)) * sum(sum(W1 ** 2)) + sum(sum(W2 ** 2)))

        return cost
