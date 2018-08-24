import numpy as np


class MultinomialNaiveBayesClassifier:

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

        # Precompute conditional probabilities
        self.conditional_probs = {}
        for classe in self.classes:
            n_total = sum(sum(self.X[self.y == classe, :]))
            for feature in range(self.n_features):
                n_samples = sum(self.X[self.y == classe, feature])
                prob = (n_samples + self.alpha) / (n_total + self.alpha * self.n_features)
                self.conditional_probs[classe, feature] = prob

    def predict(self, X):
        n_samples_test = X.shape[0]
        y_pred = np.zeros(n_samples_test)

        for i in range(n_samples_test):
            posteriori_probs = np.ones(self.n_classes)
            for classe in self.classes:
                for feature in range(self.n_features):
                    if X[i, feature] > 0:
                        prob = self.conditional_probs[classe, feature]
                        posteriori_probs[classe] *= prob

                # A posteriori probability
                posteriori_probs[classe] *= self.priori_probs[classe]
            
            # Predict the class with greatest a posteriori probability
            y_pred[i] = np.argmax(posteriori_probs)
        
        return y_pred
