import numpy as np
from scipy.optimize import minimize


class ShallowNeuralNetwork:

    def __init__(self, alpha=1e-3, max_iter=1e2, tol=1e-3, 
                 n_hid=25, lambd=0, 
                 method='batch_gradient_descent'):
        self.alpha = alpha # Learn_ing rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.n_hid = n_hid # Number of neurons in the hidden layer
        self.lambd = lambd # Regularization constant
        self.method = method # Method to be used to optimize cost function

    def fit(self, X, y):
        n_in = X.shape[1]
        n_out = np.unique(y[:]).shape[0]

        size_w1 = (n_in, self.n_hid)
        size_b1 = self.n_hid
        size_w2 = (self.n_hid, n_out)
        size_b2 = n_out
        
        self.w1 = np.random.uniform(size=size_w1)
        self.b1 = np.random.uniform(size=size_b1)
        self.w2 = np.random.uniform(size=size_w2)
        self.b2 = np.random.uniform(size=size_b2)

        if self.method == 'batch_gradient_descent':
            for _ in range(self.max_iter):
                params = self._zip_params(self.w1, self.b1, self.w2, self.b2)
                
                d = self._gradient(params, X, y, self.lambd, 
                                   n_in, self.n_hid, n_out)
                dw1, db1, dw2, db2 = self._unzip_params(d, n_in, 
                                                        self.n_hid, n_out)
                
                self.w1 -= self.alpha * dw1
                self.b1 -= self.alpha * db1
                self.w2 -= self.alpha * dw2
                self.b2 -= self.alpha * db2
                
                cost = self._cost(params, X, y, self.lambd, n_in, 
                                  self.n_hid, n_out)

                if cost <= self.tol:
                    break

        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd, n_in, 
                    self.n_hid, n_out)
            params = self._zip_params(self.w1, self.b1, self.w2, self.b2)
            res = minimize(self._cost, params, 
                           jac=self._gradient, args=args, 
                           method=self.method, options=options)
            params = res.x
            w1, b1, w2, b2 = self._unzip_params(params, n_in, self.n_hid, n_out)
            self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2

        return self

    def _zip_params(self, w1, b1, w2, b2):
        return np.concatenate((w1.reshape(-1), b1.reshape(-1), 
                               w2.reshape(-1), b2.reshape(-1)))
    
    def _unzip_params(self, params, n_in, n_hid, n_out):
        begin_w1 = 0
        end_w1 = n_in * n_hid
        end_b1 = end_w1 + n_hid
        end_w2 = end_b1 + (n_hid * n_out)

        w1 = params[begin_w1:end_w1].reshape((n_in, n_hid))
        b1 = params[end_w1:end_b1]
        w2 = params[end_w1:].reshape((n_hid, n_out))
        b2 = params[end_w2:]
        return w1, b1, w2, b2

    def _sigmoid_gradient(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  

    def predict(self, X):   
        m = X.shape[0]
        y_pred = np.zeros((m, 1))

        for i in range(m):
            x = X[i, :]
            activations, _ = self._forward_pass(x, self.w1, self.b1, 
                                                self.w2, self.b2)
            y_pred[i] = np.argmax(activations[-1])
        return y_pred

    def predict_proba(self, X):
        m = X.shape[0]
        y_pred = []

        for i in range(m):
            x = X[i, :]
            activations, _ = self._forward_pass(x, self.w1, self.b1, 
                                                self.w2, self.b2)
            y_pred.append(activations[-1])
        return np.array(y_pred)

    def _forward_pass(self, x, w1, b1, w2, b2):
        z0, a0 = x, x
        z1 = a0 @ w1 + b1
        a1 = self._sigmoid(z1)
        z2 = a1 @ w2 + b2
        a2 = self._sigmoid(z2)
        return [a0, a1, a2], [z0, z1, z2]

    def _gradient(self, params, X, y, lambd, n_in, 
                  n_hid, n_out):
        w1, b1, w2, b2 = self._unzip_params(params, n_in, n_hid, n_out)
        m = X.shape[0]
        Delta1, Delta2 = 0, 0
        for i in range(m):
            x = X[i, :]
            # Forward pass
            activations, inputs = self._forward_pass(x, w1, b1, w2, b2)

            y_pred = activations[-1]

            y_true = np.zeros(n_out)
            y_true[y[i]] = 1

            # Backward pass
            delta3 = (y_pred - y_true)

            delta2 = w2 @ delta3 + b2 * self._sigmoid_gradient(inputs[1])
            delta2 = delta2[1:]

            Delta2 += delta3 * activations[1][np.newaxis].T
            Delta1 += delta2 * activations[0][np.newaxis].T

            delta3b = 0
            deltab2 = 0

            Deltab1 += 0
            Deltab1 += 0

        dw1 = (1 / m) * Delta1
        dw2 = (1 / m) * Delta2
        db1 = (1 / m) * Deltab1
        db2 = (1 / m) * Deltab2

        dw1[:, 1:] += (lambd / m) * w1[:, 1:]
        dw2[:, 1:] += (lambd / m) * w2[:, 1:]

        return self._zip_params(dw1, db1, dw2, db2)

    def _cost(self, params, X, y, lambd, n_in, 
              n_hid, n_out):
        w1, b1, w2, b2 = self._unzip_params(params, n_in, n_hid, n_out)
        m = X.shape[0]
        cost = 0
        for i in range(m):
            x = X[i, :]
            activations, _ = self._forward_pass(x, w1, b1, w2, b2)

            y_pred = activations[-1]
            
            y_true = np.zeros(n_out)
            y_true[y[i]] = 1

            cost += -y_true @ np.log(y_pred) - (1 - y_true) @ np.log(1 - y_pred)
        cost /= m

        cost += ((lambd / (2 * m)) * 
                 sum(sum(w1[:, 1:n_in + 1] ** 2)) + 
                 sum(sum(w2[:, 1:n_hid + 1] ** 2)))

        return cost
