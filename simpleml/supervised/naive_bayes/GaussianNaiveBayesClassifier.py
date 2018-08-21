import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class GaussianNaiveBayesClassifier:

    def __init__(self, alpha=1):
        self.alpha = alpha # Laplace smoothing parameter

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
        
        # Precompute mus and sigmas to numeric features and
        # precompute conditional probabilities to categorical features
        self.mu = {}
        self.sigma = {}
        self.conditional_probs = {}
        for classe in self.classes:
            for feature in range(self.n_features):
                if self.X[:, feature].dtype == 'float64':
                    idx = self.y == classe
                    self.mu[classe, feature] = np.mean(X[idx, feature])
                    self.sigma[classe, feature] = np.std(X[idx, feature])
                elif self.X[:, feature].dtype == 'int64':
                    for value in np.unique(self.X[:, feature]):
                        idx = self.X[:, feature] == value
                        n_samples = sum(self.y[idx] == classe)
                        prob = (n_samples + self.alpha) / (sum(self.y == classe) + self.alpha * self.n_classes)
                        self.conditional_probs[classe, feature, value] = prob                

    def predict(self, X):
        n_samples_test = X.shape[0]
        y_pred = np.zeros(n_samples_test)

        for i in range(n_samples_test):
            posteriori_probs = np.ones(self.n_classes)
            for classe in self.classes:
                for feature in range(self.n_features):
                    x = X[i, feature]
                    # Conditional probability
                    if X[:, feature].dtype == 'float64':
                        mu = self.mu[classe, feature]
                        sigma = self.sigma[classe, feature]
                        prob = self._gaussian(x, mu, sigma)
                    elif X[:, feature].dtype == 'int64':
                        prob = self.conditional_probs[classe, feature, x]
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

    # Create artificial categorical feature
    X[:, 0] *= np.random.randint(10, size=X.shape[0])
    X[:, 0] = abs(X[:, 0].astype(np.int64))

    model = GaussianNaiveBayesClassifier()
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))
    