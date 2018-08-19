import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, neurons_hidden_layer=10):
        self.neurons_hidden_layer = neurons_hidden_layer

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.X = np.hstack((np.ones(self.n_samples, 1), X))
        self.y = y

        self.neurons_input_layer = self.n_features
        self.neurons_output_layer = np.unique(y[:])

    def predict(self, X):
        n_samples_test = X.shape[0]
        y_pred = np.zeros(n_samples_test)
        for i in range(n_samples_test):
            x = X[i, :]

            x = np.hstack(([1], x))
            h1 = self.sigmoid(x @ self.theta1)

            h = np.hstack(([1], h))
            h2 = self.sigmoid(h1 @ self.theta2)

            y_pred[i] = np.argmax(h2)
        return y_pred

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_gradient(self, z):
        return self._sigmoid(z) * (1 - self.sigmoid(z))
    

if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    model = NeuralNetwork()
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))
