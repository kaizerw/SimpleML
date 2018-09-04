import numpy as np


class BernoulliNaiveBayesClassifier:

    def __init__(self, alpha=1, binarize=None):
        self.alpha = alpha # Laplace smoothing parameter
        self.binarize = binarize # Threshold for binarizing features

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)

        if self.binarize:
            self.X = np.where(self.X >= 0.5, 1, 0)

        # A priori probabilities
        self.priori_probs = {}
        for classe in self.classes:
            n_samples_classe = sum(y == classe)
            self.priori_probs[classe] = n_samples_classe / self.n_samples

        # Precompute conditional probabilities
        self.conditional_probs = {}
        for classe in self.classes:
            for feature in range(self.n_features):
                idx = self.X[:, feature] == 1
                n_samples = sum(self.y[idx] == classe)
                prob = ((n_samples + self.alpha) / 
                        (sum(self.y == classe) + self.alpha * self.n_classes))
                self.conditional_probs[classe, feature] = prob

    def predict(self, X):
        n_samples_test = X.shape[0]
        y_pred = np.zeros(n_samples_test)

        for i in range(n_samples_test):
            posteriori_probs = np.ones(self.n_classes)
            for classe in self.classes:
                for feature in range(self.n_features):
                    x = X[i, feature]
                    prob = self.conditional_probs[classe, feature]
                    prob = (x * prob) + ((1 - x) * (1 - prob))
                    posteriori_probs[classe] *= prob

                # A posteriori probability
                posteriori_probs[classe] *= self.priori_probs[classe]

            # Predict the class with greatest a posteriori probability
            y_pred[i] = np.argmax(posteriori_probs)

        return y_pred

    def predict_proba(self, X):
        n_samples_test = X.shape[0]
        y_pred = []

        for i in range(n_samples_test):
            posteriori_probs = np.ones(self.n_classes)
            for classe in self.classes:
                for feature in range(self.n_features):
                    x = X[i, feature]
                    prob = self.conditional_probs[classe, feature]
                    prob = (x * prob) + ((1 - x) * (1 - prob))
                    posteriori_probs[classe] *= prob

                # A posteriori probability
                posteriori_probs[classe] *= self.priori_probs[classe]

            y_pred.append(posteriori_probs)

        return np.array(y_pred)
