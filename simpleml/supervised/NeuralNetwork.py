import numpy as np
from scipy.optimize import minimize


class NeuralNetwork:

    def __init__(self, alpha=1e-3, max_iter=1e2, tol=1e-3, 
                 neurons_hidden_layer=25, lambd=0, 
                 method='batch_gradient_descent'):
        self.alpha = alpha # Learning rate
        self.max_iter = int(max_iter) # Max iterations
        self.tol = tol # Error tolerance
        self.neurons_hidden_layer = neurons_hidden_layer # Number of neurons in 
                                                         # the hidden layer
        self.lambd = lambd # Regularization constant
        self.method = method # Method to be used to optimize cost function

    def fit(self, X, y):
        neurons_input_layer = X.shape[1]
        neurons_output_layer = np.unique(y[:]).shape[0]

        size_theta1 = (neurons_input_layer + 1, self.neurons_hidden_layer)
        size_theta2 = (self.neurons_hidden_layer + 1, neurons_output_layer)
        self.theta1 = np.random.uniform(size=size_theta1)
        self.theta2 = np.random.uniform(size=size_theta2)

        self.costs = []

        if self.method == 'batch_gradient_descent':
            for _ in range(self.max_iter):
                thetas = self._zip_thetas(self.theta1, self.theta2)
                
                theta_grads = self._gradient(thetas, X, y, self.lambd, 
                                             neurons_input_layer, 
                                             self.neurons_hidden_layer, 
                                             neurons_output_layer)
                theta1_grad, theta2_grad = self._unzip_thetas(theta_grads, 
                                             neurons_input_layer, 
                                             self.neurons_hidden_layer, 
                                             neurons_output_layer)
                
                self.theta1 -= self.alpha * theta1_grad
                self.theta2 -= self.alpha * theta2_grad
                
                cost = self._cost(thetas, X, y, self.lambd, neurons_input_layer, 
                                  self.neurons_hidden_layer, 
                                  neurons_output_layer)
                self.costs.append(cost)

                if cost <= self.tol:
                    break

        else:
            options = {'gtol': self.tol, 'maxiter': self.max_iter}
            args = (X, y, self.lambd, neurons_input_layer, 
                    self.neurons_hidden_layer, neurons_output_layer)
            thetas = self._zip_thetas(self.theta1, self.theta2)
            res = minimize(self._cost, thetas, 
                           jac=self._gradient, args=args, 
                           method=self.method, options=options)
            thetas = res.x
            t1, t2 = self._unzip_thetas(thetas, neurons_input_layer, 
                                        self.neurons_hidden_layer, 
                                        neurons_output_layer)
            self.theta1, self.theta2 = t1, t2

        return self

    def _zip_thetas(self, theta1, theta2):
        return np.concatenate((theta1.reshape(-1), theta2.reshape(-1)))
    
    def _unzip_thetas(self, thetas, neurons_input_layer, neurons_hidden_layer, 
                      neurons_output_layer):
        begin_theta1 = 0
        end_theta1 = (neurons_input_layer + 1) * neurons_hidden_layer
        theta1 = thetas[begin_theta1:end_theta1]
        theta1 = theta1.reshape((neurons_input_layer + 1, neurons_hidden_layer))
        theta2 = thetas[end_theta1:]
        theta2 = theta2.reshape((neurons_hidden_layer + 1, neurons_output_layer))
        return theta1, theta2

    def _sigmoid_gradient(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))  

    def predict(self, X):   
        n_samples = X.shape[0]
        y_pred = np.zeros((n_samples, 1))

        for i in range(n_samples):
            x = X[i, :]
            activations, _ = self._forward_pass(x, self.theta1, self.theta2)
            y_pred[i] = np.argmax(activations[-1])
        return y_pred

    def _forward_pass(self, x, theta1, theta2):
        input_l0 = x
        activation_l0 = np.hstack((1, x))

        input_l1 = activation_l0 @ theta1
        activation_l1 = np.hstack((1, self._sigmoid(input_l1)))
        
        input_l2 = activation_l1 @ theta2
        activation_l2 = self._sigmoid(input_l2)

        activations = [activation_l0, activation_l1, activation_l2]
        inputs = [input_l0, input_l1, input_l2]

        return activations, inputs

    def _gradient(self, thetas, X, y, lambd, neurons_input_layer, 
                  neurons_hidden_layer, neurons_output_layer):
        theta1, theta2 = self._unzip_thetas(thetas, neurons_input_layer, 
                                            neurons_hidden_layer, 
                                            neurons_output_layer)
        n_samples = X.shape[0]
        Delta1 = 0
        Delta2 = 0
        for i in range(n_samples):
            x = X[i, :]
            # Forward pass
            activations, inputs = self._forward_pass(x, theta1, theta2)

            y_pred = activations[-1]

            y_true = np.zeros(neurons_output_layer)
            y_true[y[i]] = 1

            # Backward pass
            delta3 = (y_pred - y_true)

            delta2 = theta2 @ delta3 * \
                     self._sigmoid_gradient(np.hstack((1, inputs[1])))
            delta2 = delta2[1:]

            Delta2 += delta3 * activations[1][np.newaxis].T
            Delta1 += delta2 * activations[0][np.newaxis].T

        Theta1_grad = (1 / n_samples) * Delta1
        Theta2_grad = (1 / n_samples) * Delta2

        Theta1_grad[:, 1:] += (lambd / n_samples) * theta1[:, 1:]
        Theta2_grad[:, 1:] += (lambd / n_samples) * theta2[:, 1:]

        return self._zip_thetas(Theta1_grad, Theta2_grad)

    def _cost(self, thetas, X, y, lambd, neurons_input_layer, 
              neurons_hidden_layer, neurons_output_layer):
        theta1, theta2 = self._unzip_thetas(thetas, neurons_input_layer, 
                                            neurons_hidden_layer, 
                                            neurons_output_layer)
        n_samples = X.shape[0]
        cost = 0
        for i in range(n_samples):
            x = X[i, :]
            activations, _ = self._forward_pass(x, theta1, theta2)

            y_pred = activations[-1]
            
            y_true = np.zeros(neurons_output_layer)
            y_true[y[i]] = 1

            cost += -y_true @ np.log(y_pred) - (1 - y_true) @ np.log(1 - y_pred)
        cost /= n_samples

        cost += ((lambd / (2 * n_samples)) * 
                 sum(sum(theta1[:, 1:neurons_input_layer + 1] ** 2)) + 
                 sum(sum(theta2[:, 1:neurons_hidden_layer + 1] ** 2)))

        return cost
