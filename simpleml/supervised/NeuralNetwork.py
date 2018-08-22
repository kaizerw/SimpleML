import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


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

        self.theta1 = np.random.random((self.n_features + 1, self.neurons_hidden_layer))
        self.theta2 = np.random.random((self.neurons_hidden_layer + 1, self.neurons_output_layer))

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
        h1 = self._sigmoid(X @ self.theta1)

        h1 = np.hstack((np.ones((n_samples_test, 1)), h1))
        h2 = self._sigmoid(h1 @ self.theta2)

        return np.argmax(h2, axis=1)
        
    def _cost(self):
        cost = 0
        for i in range(self.n_samples):
            a1 = self.X[i, :]

            a1 = np.hstack((1, a1))
            a2 = self._sigmoid(a1 @ self.theta1)
            
            a2 = np.hstack((1, a2))
            a3 = self._sigmoid(a2 @ self.theta2)
            h = a3
            
            yi = np.zeros(self.neurons_output_layer)
            yi[self.y[i]] = 1

            cost += -yi @ np.log(h) - (1 - yi) @ np.log(1 - h)
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
            a1 = self.X[i, :]

            # Feedforward pass
            a1 = np.hstack((1, a1))
            z2 = a1 @ self.theta1
            a2 = self._sigmoid(z2)
            
            a2 = np.hstack((1, a2))
            z3 = a2 @ self.theta2
            a3 = self._sigmoid(z3)

            h = a3

            yi = np.zeros(self.neurons_output_layer)
            yi[self.y[i]] = 1

            # Backward pass
            delta3 = (h - yi)
            
            delta2 = self.theta2 @ delta3 * self._sigmoid_gradient(np.hstack((1, z2)))
            delta2 = delta2[1:]

            Delta2 += delta3 * a2[np.newaxis].T
            Delta1 += delta2 * a1[np.newaxis].T

            Theta1_grad = (1 / self.n_samples) * Delta1
            Theta2_grad = (1 / self.n_samples) * Delta2

            Theta1_grad[:, 1:] += (self.lambd / self.n_samples) * self.theta1[:, 1:]
            Theta2_grad[:, 1:] += (self.lambd / self.n_samples) * self.theta2[:, 1:]

        return Theta1_grad, Theta2_grad
    

if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    model = NeuralNetwork()
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))


    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(25,), solver='sgd', learning_rate='constant', 
                          activation='logistic', learning_rate_init=1e-3, max_iter=int(1e3), alpha=0.0)
    model.fit(X, y)
    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))    
