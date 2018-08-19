import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class NaiveBayesClassifier:

    def __init__(self, alpha=1):
        self.alpha = alpha # Laplace smoothing parameter

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)

    def predict(self, X):
        n_samples_test = X.shape[0]

        # predicted classes
        y_pred = np.zeros(n_samples_test)

        # A priori probability
        priori_probs = {}
        for classe in self.classes:
            priori_probs[classe] = sum(y == classe) / self.n_samples

        for i in range(n_samples_test):
            posteriori_probs = {}
            for classe in self.classes:
                posteriori_probs[classe] = 1

                for feature in range(self.n_features):
                    idx = self.X[:, feature] == X[i, feature]
                    samples = sum(self.y[idx] == classe)
                    # Conditional probability
                    prob = (samples + self.alpha) / (sum(self.y == classe) + self.alpha * self.n_classes)
                    posteriori_probs[classe] *= prob

                # A posteriori probability
                posteriori_probs[classe] *= priori_probs[classe]

            # Predict the class with greatest a posteriori probability
            max_prob = -np.inf
            for classe in self.classes:
                prob = posteriori_probs[classe]
                if prob > max_prob:
                    max_prob = prob
                    y_pred[i] = classe
        
        return y_pred


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    # By now, only works with categorical features
    # TODO: Generalize to numeric and mixed features
    X = abs(X.astype(int))

    model = NaiveBayesClassifier()
    model.fit(X, y)
    
    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))
