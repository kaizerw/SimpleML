import numpy as np
from scipy.optimize import minimize


class DeepNeuralNetwork:

    def __init__(self, alpha=1e-5, max_iter=1e4, tol=1e-3, n_hid=(10, 5), 
                 lambd=0, beta1=0.9, beta2=0.999, activation='relu', 
                 method='batch_gradient_descent'):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.n_hid = n_hid # Number of neurons in hidden layers
        self.lambd = lambd # Regularization constant
        self.beta1 = beta1 # Momentum constant
        self.beta2 = beta2 # RMSprop constant
        self.method = method # Method to be used to optimize cost function
        if activation == 'sigmoid':
            self.activation_hidden = self.__sigmoid
            self.gradient_hidden = self.__sigmoid_gradient
        elif activation == 'tanh':
            self.activation_hidden = self.__tanh
            self.gradient_hidden = self.__tanh_gradient
        elif activation == 'relu':
            self.activation_hidden = self.__relu
            self.gradient_hidden = self.__relu_gradient

    def fit(self, X, y):
        if len(np.unique(y)) > 2:
            raise Exception('This classifier only works with two classes')

        y = y.reshape((-1, 1))

        n_in, n_out = X.shape[1], 1
        self.n_layers = []
        self.n_layers.append(n_in)
        self.n_layers.extend(self.n_hid)
        self.n_layers.append(n_out)
        self.L = len(self.n_layers) - 1 # Do not count input layer

        self.W, self.b = {}, {}
        for l in range(1, self.L + 1):
            # He initialization
            self.W[l] = np.random.randn(self.n_layers[l - 1], self.n_layers[l]) * np.sqrt(2 / self.n_layers[l - 1])
            self.b[l] = np.zeros(self.n_layers[l])

        self.activations = {}
        self.gradients = {}
        for l in range(1, self.L):
            self.activations[l] = self.activation_hidden
            self.gradients[l] = self.gradient_hidden
        self.activations[self.L] = self.__sigmoid
        self.gradients[self.L] = self.__sigmoid_gradient

        if self.method == 'batch_gradient_descent':
            VdW, Vdb, VdW_c, Vdb_c = {}, {}, {}, {}
            SdW, Sdb, SdW_c, Sdb_c = {}, {}, {}, {}
            for l in range(1, self.L + 1):
                VdW[l] = np.zeros(self.W[l].shape)
                Vdb[l] = np.zeros(self.b[l].shape)
                SdW[l] = np.zeros(self.W[l].shape)
                Sdb[l] = np.zeros(self.b[l].shape)

            for t in range(1, self.max_iter + 1):
                params = self.__zip_params(self.W, self.b, self.L)
                
                d = self.__gradient(params, X, y, self.n_layers, self.L, self.lambd)
                dW, db = self.__unzip_params(d, self.n_layers, self.L)
                
                for l in range(1, self.L + 1):
                    # Momentum
                    VdW[l] = self.beta1 * VdW[l] + (1 - self.beta1) * dW[l]
                    Vdb[l] = self.beta1 * Vdb[l] + (1 - self.beta1) * db[l]

                    VdW_c[l] = VdW[l] / (1 - self.beta1 ** t)
                    Vdb_c[l] = Vdb[l] / (1 - self.beta1 ** t)

                    # RMSprop
                    SdW[l] = self.beta2 * SdW[l] + (1 - self.beta2) * dW[l] ** 2
                    Sdb[l] = self.beta2 * Sdb[l] + (1 - self.beta2) * db[l] ** 2

                    SdW_c[l] = SdW[l] / (1 - self.beta2 ** t)
                    Sdb_c[l] = Sdb[l] / (1 - self.beta2 ** t)

                    self.W[l] -= self.alpha * (VdW_c[l] / (np.sqrt(SdW_c[l] + 1e-8)))
                    self.b[l] -= self.alpha * (Vdb_c[l] / (np.sqrt(Sdb_c[l] + 1e-8)))
                
                cost = self.__cost(params, X, y, self.n_layers, self.L, self.lambd)

                if cost <= self.tol:
                    break

        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.n_layers, self.L, self.lambd)
            params = self.__zip_params(self.W, self.b, self.L)
            res = minimize(self.__cost, params, jac=self.__gradient, args=args, 
                           method=self.method, options=options)
            params = res.x
            W, b = self.__unzip_params(params, self.n_layers, self.L)
            self.W, self.b = W, b

        return self

    def __zip_params(self, W, b, L):
        params = []
        for l in range(1, L + 1):
            params.append(W[l].reshape(-1))
            params.append(b[l].reshape(-1))
        return np.concatenate(params)
    
    def __unzip_params(self, params, n_layers, L):
        desl = 0
        W, b = {}, {}
        for l in range(1, L + 1):
            m, n = n_layers[l - 1], n_layers[l]
            begin_w = desl
            end_w = begin_w + m * n
            begin_b = end_w
            end_b = begin_b + n
            desl = end_b
            W[l] = params[begin_w:end_w].reshape((m, n))
            b[l] = params[begin_b:end_b]

        return W, b

    def __linear(self, z):
        return z

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
        _, As = self.__forward_pass(X, self.W, self.b, self.L)
        return np.where(As[-1] > 0.5, 1, 0).reshape(-1)

    def predict_proba(self, X):
        _, As = self.__forward_pass(X, self.W, self.b, self.L)
        return As[-1].reshape(-1)

    def __forward_pass(self, X, W, b, L):
        Zs, As = [X], [X]

        for l in range(1, L + 1):
            Z = As[l - 1] @ W[l] + b[l]
            A = self.activations[l](Z)
            Zs.append(Z)
            As.append(A)
        
        return Zs, As

    def __backward_pass(self, Zs, As, W, L, X, y, lambd):
        m = X.shape[0]

        dA = As[-1] - y

        dW, db = {}, {}
        for l in range(L, 0, -1):
            dZ = dA * self.gradients[l](Zs[l])

            dW[l] = (1 / m) * As[l - 1].T @ dZ
            dW[l] += (lambd / m) * W[l]
            db[l] = (1 / m) * np.sum(dZ, axis=0)
            dA = dZ @ W[l].T
        
        return dW, db

    def __gradient(self, params, X, y, n_layers, L, lambd):
        W, b = self.__unzip_params(params, n_layers, L)

        Zs, As = self.__forward_pass(X, W, b, L)
        dW, db = self.__backward_pass(Zs, As, W, L, X, y, lambd)

        return self.__zip_params(dW, db, L)

    def __cost(self, params, X, y, n_layers, L, lambd):
        W, b = self.__unzip_params(params, n_layers, L)
        m = X.shape[0]

        _, As = self.__forward_pass(X, W, b, L)
        A_last = As[-1]
        cost = np.sum(y * np.log(A_last) + (1 - y) * np.log(1 - A_last))
        cost *= -(1 / m)
        for l in range(1, L + 1):
            cost += ((lambd / (2 * m))) * sum(sum(W[l] ** 2))
        return cost
