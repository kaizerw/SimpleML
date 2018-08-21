import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, neurons_hidden_layer=25, lambd=0):
        self.neurons_hidden_layer = neurons_hidden_layer
        self.lambd = lambd

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = X
        self.y = y

        self.neurons_input_layer = self.n_features
        self.neurons_output_layer = np.unique(y[:]).shape[0]

        self.theta1 = np.random.random((self.n_features + 1, self.neurons_hidden_layer))
        self.theta2 = np.random.random((self.neurons_hidden_layer + 1, self.neurons_output_layer))

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

            yi = np.zeros(self.neurons_output_layer)
            yi[self.y[i]] = 1

            # Backward pass
            delta3 = (a3 - yi)
            
            delta2 = self.theta2 @ delta3 * self._sigmoid_gradient(np.hstack((1, z2)))
            delta2 = delta2[1:]

            Delta2 += delta3 * a2
            Delta1 += delta2 * a1

        Theta1_grad = (1 / self.n_samples) * Delta1
        Theta2_grad = (1 / self.n_samples) * Delta2

        Theta1_grad[:, 1:] += (self.lambd / self.n_samples) * self.theta1[:, 1:]
        Theta2_grad[:, 1:] += (self.lambd / self.n_samples) * self.theta2[:, 1:]
            

    def predict(self, X):
        n_samples_test = X.shape[0]

        X = np.hstack((np.ones((n_samples_test, 1)), X))
        # [n_samples_test, n_features + 1] 
        # @ [n_features + 1, neurons_hidden_layer] 
        # = [n_samples_test, neurons_hidden_layer]
        h1 = self._sigmoid(X @ self.theta1)
        h1 = np.hstack((np.ones((n_samples_test, 1)), h1))
        # [n_samples_test, neurons_hidden_layer + 1] 
        # @ [neurons_hidden_layer + 1, neurons_output_layer] 
        # = [n_samples_test, neurons_output_layer]
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

            cost += -yi * np.log(h) - (1 - yi) * np.log(1 - h)
        cost /= self.n_samples

        cost += ((self.lambd / (2 * self.n_samples)) * 
                 sum(sum(self.theta1[:, 1:self.neurons_input_layer + 1] ** 2)) + 
                 sum(sum(self.theta2[:, 1:self.neurons_hidden_layer + 1] ** 2)))
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_gradient(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))
    

if __name__ == '__main__':
    X, y = make_classification(n_samples=5000, n_features=400, n_informative=400, 
                               n_redundant=0, n_repeated=0, n_classes=10)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    model = NeuralNetwork()
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))
