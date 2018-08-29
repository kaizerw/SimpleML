import numpy as np
from scipy.stats import mode
from random import choices, sample


class DecisionTreeClassifier:

    def __init__(self, criterion='information_gain', max_depth=100, 
                 num_cuts=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.num_cuts = num_cuts
        if self.num_cuts is not None:
            self.num_cuts = int(self.num_cuts)
        self._eval_criterion = {}
        if criterion == 'information_gain':
            self._eval_criterion = self._information_gain
        elif criterion == 'gain_ratio':
            self._eval_criterion = self._gain_ratio
        elif criterion == 'gini_index':
            self._eval_criterion = self._gini_index

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.X = X
        self.y = y

        samples_idx = np.array(range(self.n_samples))
        k = int(np.sqrt(self.n_features))
        features_idx = sample(list(range(self.n_features)), k=k)
        self._tree = self._step(samples_idx, features_idx, 0)

    def _step(self, samples_idx, features_idx, depth):
        node = {}

        # If all samples in 'samples_idx' have a same label 'y', 
        # return a leave node labeled with 'y'
        if self._all_same_class(samples_idx):
            node['prediction'] = self.y[samples_idx[0]]
            node['is_leave'] = True
            return node

        # If the list of features is empty or depth equals to max depth
        # return a leave node labeled with the most 
        # frequent class in 'samples_idx'
        if len(features_idx) == 0 or depth == self.max_depth:
            node['prediction'] = self._most_frequent_class(samples_idx)
            node['is_leave'] = True
            return node

        # Select feature with best criterion division
        best_feature = self._choose_best_feature(samples_idx, features_idx)
        
        # Assign best feature to new node
        node['feature'] = best_feature

        # Remove best feature from the list of features 'features_idx'
        new_features_idx = list(features_idx)
        new_features_idx.remove(best_feature)
        new_features_idx = np.array(new_features_idx)

        node['children'] = []
        # Collect cut points of 'best_feature' feature, 
        # considering only the samples in 'samples_idx'
        cuts, kind = self._select_cut_points(samples_idx, best_feature)
        # For each cut point...
        for value in cuts:
            if kind == 'categorical':
                # Collect only samples with 'best_feature' value 
                # equal to 'value'
                new_samples_idx = self.X[samples_idx, best_feature] == value
                new_samples_idx = samples_idx[new_samples_idx]
            elif kind == 'numeric':
                # Collect only samples with 'best_feature' value 
                # less than 'value'
                new_samples_idx = self.X[samples_idx, best_feature] < value
                new_samples_idx = samples_idx[new_samples_idx]
            
            # If 'new_samples_idx' is empty, return a leave node labeled 
            # with the most frequent class in 'new_samples_idx'
            if len(new_samples_idx) == 0:
                node['prediction'] = self._most_frequent_class(new_samples_idx)
                node['is_leave'] = True
                return node

            # Create recursively one subtree for each cut point
            # of 'best_feature' feature 
            node['is_leave'] = False
            child = {'value': value,
                     'feature': best_feature,
                     'kind': kind, 
                     'is_leave': False, 
                     'tree': self._step(new_samples_idx, new_features_idx, 
                                        depth + 1)}
            node['children'].append(child)

        return node

    def predict(self, x):
        y_pred = np.zeros(x.shape[0])
        i = 0
        node = self._tree
        while i < x.shape[0]:
            if node['is_leave']:
                y_pred[i] = node['prediction']
                i += 1
                node = self._tree
            else:
                for child in node['children']:
                    feature = child['feature']
                    value = child['value']
                    kind = child['kind']
                    if ((kind == 'categorical' and x[i, feature] == value) or 
                       (kind == 'numeric' and x[i, feature] < value)):
                        if child['is_leave']:
                            y_pred[i] = child['prediction']
                            i += 1
                            node = self._tree
                        else:
                            node = child['tree']
                        break
        return y_pred

    def _all_same_class(self, samples_idx):
        for classe in self.classes:
            count = sum(self.y[samples_idx] == classe)
            if count == samples_idx.shape[0]:
                return True
        return False

    def _most_frequent_class(self, samples_idx):
        clas = None
        max_samples = -np.inf
        for classe in self.classes:
            count = sum(self.y[samples_idx] == classe)
            if count > max_samples:
                max_samples = count
                clas = classe
        return clas

    def _choose_best_feature(self, samples_idx, features_idx):
        best_feature = None
        max_gain = -np.inf
        for feature in features_idx:
            gain = self._eval_criterion(samples_idx, feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        return best_feature

    def _select_cut_points(self, samples_idx, feature):
        cuts = []
        kind = ''
        if self.X[: feature].dtype == 'int64':
            # Select all distinct values of this features as cuts
            cuts = np.unique(self.X[samples_idx, feature])
            kind = 'categorical'
        elif self.X[:, feature].dtype == 'float64':
            if self.num_cuts is None:
                data = []
                for i in samples_idx:
                    data.append([self.X[i, feature], self.y[i]])
                
                data.sort(key=lambda i: i[0])

                last_class = data[0][1]
                # Select all point cuts of this feature
                # In this case, a cut point is the mean of the two values
                # where a change of class occurs
                for i in range(1, len(data)):
                    if data[i][1] != last_class: 
                        cuts.append(np.mean((data[i - 1][0], data[i][0])))
                    last_class = data[i][1]
                cuts.append(np.max(self.y[samples_idx]) + 1)
            else:
                min_x = self.X[:, feature].min()
                max_x = self.X[:, feature].max()
                cuts = list(np.linspace(min_x, max_x + 1, self.num_cuts))
            kind = 'numeric'

        return cuts, kind

    def _entropy(self, samples_idx, feature=None):
        info = 0
        n_total = len(samples_idx)
        
        if feature is None:
            for classe in self.classes:
                n_classe = sum(self.y[samples_idx] == classe)
                prob_classe = n_classe / n_total
                if prob_classe > 0:
                    info += prob_classe * np.log2(prob_classe)
            info = -info
        else:
            values, kind = self._select_cut_points(samples_idx, feature)
            for value in values:
                if kind == 'categorical':
                    n_value = sum(self.X[samples_idx, feature] == value)
                    idx = np.where(self.X[samples_idx, feature] == value)
                elif kind == 'numeric':
                    n_value = sum(self.X[samples_idx, feature] < value)
                    idx = np.where(self.X[samples_idx, feature] < value)
                prob_value = n_value / n_total
                if prob_value > 0:
                    feature_samples_idx = samples_idx[idx]
                    info += prob_value * self._entropy(feature_samples_idx)
        
        return info

    def _split_info(self, samples_idx, feature):
        info = 0
        n_total = len(samples_idx)
        
        values, kind = self._select_cut_points(samples_idx, feature)
        for value in values:
            if kind == 'categorical':
                n_value = sum(self.X[samples_idx, feature] == value)
            elif kind == 'numeric':
                n_value = sum(self.X[samples_idx, feature] < value)    
            prob_value = n_value / n_total
            if prob_value > 0:
                info += prob_value * np.log2(prob_value)
        
        return -info

    def _information_gain(self, samples_idx, feature):
        info_all = self._entropy(samples_idx)
        info_feature = self._entropy(samples_idx, feature)
        return info_all - info_feature

    def _gain_ratio(self, samples_idx, feature):
        split_info = self._split_info(samples_idx, feature)
        if split_info > 0:
            information_gain = self._information_gain(samples_idx, feature)
            return information_gain / split_info
        return 0

    def _gini(self, samples_idx):
        gini = 1
        n_total = len(samples_idx)

        for _ in self.classes:
            n_samples = sum(self.y[samples_idx])
            prob_classe = n_samples / n_total
            gini -= prob_classe ** 2

        return gini

    def _gini_index(self, samples_idx, feature):
        gini_all = self._gini(samples_idx)

        ginis_feature = {}
        n_total = len(samples_idx)
        values, kind = self._select_cut_points(samples_idx, feature)
        for value in values:
            ginis_feature[value] = 0

            if kind == 'categorical':
                n_values = np.sum(self.X[samples_idx, feature] == value)
                idx = np.where(self.X[samples_idx, feature] == value)
            elif kind == 'numeric':
                n_values = np.sum(self.X[samples_idx, feature] < value)
                idx = np.where(self.X[samples_idx, feature] < value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[idx]
                aux = prob_value * self._gini(feature_samples_idx)
                ginis_feature[value] += aux

            if kind == 'categorical':
                n_values = np.sum(self.X[samples_idx, feature] != value)
                idx = np.where(self.X[samples_idx, feature] != value)
            elif kind == 'numeric':
                n_values = np.sum(self.X[samples_idx, feature] >= value)
                idx = np.where(self.X[samples_idx, feature] >= value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[idx]
                aux = prob_value * self._gini(feature_samples_idx)
                ginis_feature[value] += aux
        
        return gini_all - min(ginis_feature.values())


class RandomForestClassifier:

    def __init__(self, n_models=10):
        self.models = [DecisionTreeClassifier() 
                       for _ in range(n_models)]
        self.n_models = n_models
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        possible_idx = list(range(n_samples))
        for model in self.models:
            bag_idx = choices(possible_idx, k=n_samples)
            model.fit(X[bag_idx, :], y[bag_idx])

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = []
        for i in range(n_samples):
            x = np.reshape(X[i, :], (1, -1))
            predictions = [model.predict(x)[0] for model in self.models]
            y_pred.append(mode(predictions).mode[0])
        return np.array(y_pred)
