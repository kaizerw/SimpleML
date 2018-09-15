import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class ShallowNeuralNetwork:

    def __init__(self, alpha=1e-5, max_iter=1e4, tol=1e-3, n_hid=25, lambd=0, 
                 beta1=0.9, beta2=0.999, activation='relu', 
                 method='batch_gradient_descent', show_cost_plot=False):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.n_hid = n_hid # Number of neurons in the hidden layer
        self.lambd = lambd # Regularization constant
        self.beta1 = beta1 # Momentum constant
        self.beta2 = beta2 # RMSprop constant
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
        self.show_cost_plot = show_cost_plot # If show plot of cost function

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        y = y.reshape((-1, 1))

        n_in, n_out = X.shape[1], 1
        
        # He initialization
        self.W1 = np.random.randn(n_in, self.n_hid) * np.sqrt(2 / n_in)
        self.b1 = np.zeros(self.n_hid)
        self.W2 = np.random.randn(self.n_hid, n_out) * np.sqrt(2 / self.n_hid)
        self.b2 = np.zeros(n_out)

        if self.method == 'batch_gradient_descent':
            if self.show_cost_plot:
                plt.show()
                plt.xlabel('Iteration')
                plt.ylabel('Cost')
                axes = plt.gca()
                axes.set_xlim(0, self.max_iter)
                axes.set_ylim(0, self.__cost(self.__zip_params(self.W1, self.b1, 
                                             self.W2, self.b2), X, y, self.lambd, 
                                             n_in, self.n_hid, n_out))
                line, = axes.plot([], [], 'r-')

            VdW1 = np.zeros(self.W1.shape)
            Vdb1 = np.zeros(self.b1.shape)
            VdW2 = np.zeros(self.W2.shape)
            Vdb2 = np.zeros(self.b2.shape)

            SdW1 = np.zeros(self.W1.shape)
            Sdb1 = np.zeros(self.b1.shape)
            SdW2 = np.zeros(self.W2.shape)
            Sdb2 = np.zeros(self.b2.shape)

            for t in range(1, self.max_iter + 1):
                params = self.__zip_params(self.W1, self.b1, self.W2, self.b2)
                
                d = self.__gradient(params, X, y, self.lambd, 
                                    n_in, self.n_hid, n_out)
                dW1, db1, dW2, db2 = self.__unzip_params(d, n_in, 
                                                         self.n_hid, n_out)
                # Momentum
                VdW1 = self.beta1 * VdW1 + (1 - self.beta1) * dW1
                Vdb1 = self.beta1 * Vdb1 + (1 - self.beta1) * db1
                VdW2 = self.beta1 * VdW2 + (1 - self.beta1) * dW2
                Vdb2 = self.beta1 * Vdb2 + (1 - self.beta1) * db2

                VdW1_c = VdW1 / (1 - self.beta1 ** t)
                Vdb1_c = Vdb1 / (1 - self.beta1 ** t)
                VdW2_c = VdW2 / (1 - self.beta1 ** t)
                Vdb2_c = Vdb2 / (1 - self.beta1 ** t)

                # RMSprop
                SdW1 = self.beta2 * SdW1 + (1 - self.beta2) * dW1 ** 2
                Sdb1 = self.beta2 * Sdb1 + (1 - self.beta2) * db1 ** 2
                SdW2 = self.beta2 * SdW2 + (1 - self.beta2) * dW2 ** 2
                Sdb2 = self.beta2 * Sdb2 + (1 - self.beta2) * db2 ** 2

                SdW1_c = SdW1 / (1 - self.beta2 ** t)
                Sdb1_c = Sdb1 / (1 - self.beta2 ** t)
                SdW2_c = SdW2 / (1 - self.beta2 ** t)
                Sdb2_c = Sdb2 / (1 - self.beta2 ** t)

                self.W1 -= self.alpha * (VdW1_c / (np.sqrt(SdW1_c) + 1e-8))
                self.b1 -= self.alpha * (Vdb1_c / (np.sqrt(Sdb1_c) + 1e-8))
                self.W2 -= self.alpha * (VdW2_c / (np.sqrt(SdW2_c) + 1e-8))
                self.b2 -= self.alpha * (Vdb2_c / (np.sqrt(Sdb2_c) + 1e-8))
                
                cost = self.__cost(params, X, y, self.lambd, n_in, 
                                   self.n_hid, n_out)

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
        dW1 += (lambd / m) * W1
        dW2 += (lambd / m) * W2

        return self.__zip_params(dW1, db1, dW2, db2)

    def __cost(self, params, X, y, lambd, n_in, n_hid, n_out):
        W1, b1, W2, b2 = self.__unzip_params(params, n_in, n_hid, n_out)
        m = X.shape[0]

        Z1, A1, Z2, A2 = self.__forward_pass(X, W1, b1, W2, b2)
        cost = -(1 / m) * np.sum(y * np.log(A2) + (1 - y) * np.log(1 - A2))
        cost += ((lambd / (2 * m)) * sum(sum(W1 ** 2)) + sum(sum(W2 ** 2)))

        return cost
