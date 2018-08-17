import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class NaiveBayesClassifier:

    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        X = self.X
        y = np.reshape(self.y, (self.y.shape[0], 1))

        # Target classes
        classes = np.unique(y)

        print(classes)

        # A priori probabilities
        priori_probs = {}
        for classe in classes:
            priori_probs[classe] = sum(y==classe) / y.shape[0]

        prods = {}
        for classe in classes:
            prod = 1.0

            for feature in range(x.shape[1]):
                samples = np.where(X[:, feature]==x[0, feature])
                samples = sum(y[samples]==classe)
                prod *= samples / priori_probs[classe]

            prods[classe] = priori_probs[classe] * prod

        m = -np.inf
        y_pred = -1
        for classe in classes:
            p = prods[classe]
            if p > m:
                m = p
                y_pred = classe
        return y_pred


if __name__ == '__main__':
    # TODO: Test deeply NaiveBayesClassifier

    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    # By now only works with categorical features
    # TODO: Generalize to numeric and mixed features
    X = abs(X.astype(int))

    model = NaiveBayesClassifier()
    model.fit(X, y)

    x = X[0, :].reshape((1, -1))
    y = y.reshape((-1, 1))

    print('Accuracy:', model.predict(x)==y[0])
