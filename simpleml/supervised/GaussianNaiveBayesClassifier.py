import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class GaussianNaiveBayesClassifier:

    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)

        # A priori probabilities
        self.priori_probs = {}
        for classe in self.classes:
            n_samples_classe = sum(y == classe)
            self.priori_probs[classe] = n_samples_classe / self.n_samples
        
        # Precompute mus and sigmas
        self.mu = {}
        self.sigma = {}
        for classe in self.classes:
            for feature in range(self.n_features):
                idx = self.y == classe
                self.mu[classe, feature] = np.mean(X[idx, feature])
                self.sigma[classe, feature] = np.std(X[idx, feature])

    def predict(self, X):
        n_samples_test = X.shape[0]
        y_pred = np.zeros(n_samples_test)

        for i in range(n_samples_test):
            posteriori_probs = np.ones(self.n_classes)
            for classe in self.classes:
                posteriori_probs[classe] = 1
                for feature in range(self.n_features):
                    x = X[i, feature]
                    mu = self.mu[classe, feature]
                    sigma = self.sigma[classe, feature]
                    # Conditional probability
                    prob = self._gaussian(x, mu, sigma)
                    posteriori_probs[classe] *= prob

                # A posteriori probability
                posteriori_probs[classe] *= self.priori_probs[classe]

            # Predict the class with greatest a posteriori probability
            y_pred[i] = np.argmax(posteriori_probs)

        return y_pred

    def _gaussian(self, x, mu, sigma):
        return (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(-((x - mu) ** 2 / (2 * sigma ** 2)))


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = GaussianNaiveBayesClassifier()
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))
