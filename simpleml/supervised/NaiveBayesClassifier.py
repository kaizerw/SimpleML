import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


class NaiveBayesClassifier:

    def __init__(self, alpha=1):
        self.alpha = alpha # Laplace smoothing parameter

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, x):
        X = self.X
        y = np.reshape(self.y, (self.y.shape[0], 1))

        # Target classes
        classes = np.unique(y)

        # predicted classes
        y_pred = np.zeros((x.shape[0], 1))

        # A priori probability
        priori_probs = {}
        for classe in classes:
            priori_probs[classe] = sum(y==classe) / y.shape[0]

        for i in range(x.shape[0]):
            posteriori_probs = {}
            for classe in classes:
                posteriori_probs[classe] = 1.0

                for feature in range(x.shape[1]):
                    idx = np.where(X[:, feature]==x[i, feature])
                    samples = sum(y[idx]==classe)
                    # Conditional probability
                    prob = (samples + self.alpha) / (sum(y==classe) + self.alpha * len(classes))
                    posteriori_probs[classe] *= prob

                # A posteriori probability
                posteriori_probs[classe] *= priori_probs[classe]

            # Predict the class with greatest a posteriori probability
            max_prob = -np.inf
            for classe in classes:
                prob = posteriori_probs[classe]
                if prob > max_prob:
                    max_prob = prob
                    y_pred[i, 0] = classe
        
        return y_pred


if __name__ == '__main__':
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    # By now, only works with categorical features
    # TODO: Generalize to numeric and mixed features
    X = abs(X.astype(int))

    model = NaiveBayesClassifier()
    model.fit(X, y)

    y = y.reshape((-1, 1))
        
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred==y) / y.shape[0]))
