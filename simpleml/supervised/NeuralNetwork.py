import numpy as np


class NeuralNetwork:

    def __init__(self, alpha=1e-3, max_iter=1e3, tol=1e-3, neurons_hidden_layer=25, lambd=0):
        self.alpha = alpha # Learning rate
        self.max_iter = max_iter # Max iterations
        self.tol = tol # Error tolerance
        self.neurons_hidden_layer = neurons_hidden_layer # Number of neurons in the hidden layer
        self.lambd = lambd # Regularization constant

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y

        self.neurons_input_layer = self.n_features
        self.neurons_output_layer = np.unique(y[:]).shape[0]

        self.theta1 = np.random.uniform(1e-10, 1e-20, 
                                        size=(self.n_features + 1, self.neurons_hidden_layer))
        self.theta2 = np.random.uniform(1e-10, 1e-20, 
                                        size=(self.neurons_hidden_layer + 1, self.neurons_output_layer))

        self.costs = []

        i = 0
        while True:
            theta1_grad, theta2_grad = self._gradient()
            self.theta1 -= self.alpha * theta1_grad
            self.theta2 -= self.alpha * theta2_grad
            
            cost = self._cost()
            self.costs.append(cost)

            if i >= self.max_iter or cost <= self.tol:
                break
            
            i += 1

        return self       

    def predict(self, X):
        n_samples_test = X.shape[0]

        X = np.hstack((np.ones((n_samples_test, 1)), X))
        activation_l1 = self._sigmoid(X @ self.theta1)

        activation_l1 = np.hstack((np.ones((n_samples_test, 1)), activation_l1))
        activation_l2 = self._sigmoid(activation_l1 @ self.theta2)

        return np.argmax(activation_l2, axis=1)
        
    def _cost(self):
        cost = 0
        for i in range(self.n_samples):
            activation_l0 = self.X[i, :]

            activation_l0 = np.hstack((1, activation_l0))
            activation_l1 = self._sigmoid(activation_l0 @ self.theta1)
            
            activation_l1 = np.hstack((1, activation_l1))
            activation_l2 = self._sigmoid(activation_l1 @ self.theta2)
            y_pred = activation_l2
            
            y_true = np.zeros(self.neurons_output_layer)
            y_true[self.y[i]] = 1

            cost += -y_true @ np.log(y_pred) - (1 - y_true) @ np.log(1 - y_pred)
        cost /= self.n_samples

        cost += ((self.lambd / (2 * self.n_samples)) * 
                 sum(sum(self.theta1[:, 1:self.neurons_input_layer + 1] ** 2)) + 
                 sum(sum(self.theta2[:, 1:self.neurons_hidden_layer + 1] ** 2)))

        return cost
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_gradient(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _gradient(self):
        Delta1 = 0
        Delta2 = 0
        for i in range(self.n_samples):
            activation_l0 = self.X[i, :]

            # Feedforward pass
            activation_l0 = np.hstack((1, activation_l0))
            input_l1 = activation_l0 @ self.theta1
            activation_l1 = self._sigmoid(input_l1)
            
            activation_l1 = np.hstack((1, activation_l1))
            input_l2 = activation_l1 @ self.theta2
            activation_l2 = self._sigmoid(input_l2)

            y_pred = activation_l2

            y_true = np.zeros(self.neurons_output_layer)
            y_true[self.y[i]] = 1

            # Backward pass
            delta3 = (y_pred - y_true)

            delta2 = self.theta2 @ delta3 * self._sigmoid_gradient(np.hstack((1, input_l1)))
            delta2 = delta2[1:]

            Delta2 += delta3 * activation_l1[np.newaxis].T
            Delta1 += delta2 * activation_l0[np.newaxis].T

        Theta1_grad = (1 / self.n_samples) * Delta1
        Theta2_grad = (1 / self.n_samples) * Delta2

        Theta1_grad[:, 1:] += (self.lambd / self.n_samples) * self.theta1[:, 1:]
        Theta2_grad[:, 1:] += (self.lambd / self.n_samples) * self.theta2[:, 1:]

        return Theta1_grad, Theta2_grad
