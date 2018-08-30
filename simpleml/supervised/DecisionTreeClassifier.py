import numpy as np
import operator


class DecisionTreeClassifier:

    def __init__(self, criterion='information_gain', max_depth=1000):
        self.criterion = criterion
        self.max_depth = max_depth
        self._eval = {
            'information_gain': self._information_gain, 
            'gain_ratio': self._gain_ratio, 
            'gini_index': self._gini_index
        }
        self._comps = {
            'categorical': operator.eq, 
            'numeric': operator.le
        }

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        self.X = X
        self.y = y

        samples_idx = np.array(range(self.n_samples))
        features_idx = np.array(range(self.n_features))
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

        # Remove best feature from the list of features 'features_idx'
        new_features_idx = list(features_idx)
        new_features_idx.remove(best_feature)
        new_features_idx = np.array(new_features_idx)

        # Collect cut points of 'best_feature' feature, 
        # considering only the samples in 'samples_idx'
        kind = self._get_feature_kind(best_feature)
        cuts = self._select_cut_points(samples_idx, best_feature, kind)

        node['feature'] = best_feature
        node['children'] = []
        node['kind'] = kind

        if kind == 'categorical':
            # For each distinct category value...
            for value in cuts:
                # Collect only samples with 'best_feature' value 
                # equal to 'value'
                new_samples_idx = self.X[samples_idx, best_feature] == value
                new_samples_idx = samples_idx[new_samples_idx]

                # If 'new_samples_idx' is not empty then 
                # create recursively one subtree for each distinct category 
                # value of 'best_feature' feature 
                if len(new_samples_idx) > 0:
                    node['is_leave'] = False
                    child = {'value': value,
                            'feature': best_feature,
                            'kind': kind,
                            'is_leave': False, 
                            'tree': self._step(new_samples_idx, 
                                               new_features_idx, 
                                               depth + 1)}
                    node['children'].append(child)

        elif kind == 'numeric':
            cut_point = cuts[0]

            # Collect only samples with 'best_feature' value 
            # less or equal than 'cut_point'
            idx_left = self._comps[kind](self.X[samples_idx, best_feature], cut_point)
            new_samples_idx_left = samples_idx[idx_left]

            # If 'new_samples_idx_left' is not empty then
            # create recursively one subtree for samples with 
            # 'best_feature' value <= cut point 
            if len(new_samples_idx_left) > 0:
                node['is_leave'] = False
                child = {'value': cut_point,
                        'feature': best_feature,
                        'kind': kind, 
                        'is_leave': False, 
                        'tree': self._step(new_samples_idx_left, 
                                           new_features_idx, 
                                           depth + 1)}
                node['children'].append(child)

            # Collect only samples with 'best_feature' value 
            # greater than 'cut_point'
            idx_right = ~idx_left
            new_samples_idx_right = samples_idx[idx_right]

            # If 'new_samples_idx_right' is not empty then
            # create recursively one subtree for samples with 
            # 'best_feature' value > cut point 
            if len(new_samples_idx_right) > 0:
                node['is_leave'] = False
                child = {'value': cut_point,
                        'feature': best_feature,
                        'is_leave': False, 
                        'tree': self._step(new_samples_idx_right, 
                                           new_features_idx, 
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
                kind = node['kind']
                if kind == 'categorical':
                    for child in node['children']:
                        if child['is_leave']:
                            y_pred[i] = child['prediction']
                            i += 1
                            node = self._tree
                            break

                        feature = child['feature']
                        kind = child['kind']
                        value = child['value']
                        if self._comps[kind](x[i, feature], value):
                            node = child['tree']
                            break
                elif kind == 'numeric':
                    feature = node['children'][0]['feature']
                    value = node['children'][0]['value']
                    if self._comps[kind](x[i, feature], value):
                        node = node['children'][0]['tree']
                    else:
                        node = node['children'][1]['tree']
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
            gain = self._eval[self.criterion](samples_idx, feature)
            if gain > max_gain:
                max_gain = gain
                best_feature = feature
        return best_feature

    def _get_feature_kind(self, feature):
        kind = ''
        if self.X[:, feature].dtype == 'int64':
            kind = 'categorical'
        elif self.X[:, feature].dtype == 'float64':
            kind = 'numeric'
        return kind

    def _select_cut_points(self, samples_idx, feature, kind):
        cuts = []

        if kind == 'categorical':
            # Select all distinct values of this features as cuts
            cuts = np.unique(self.X[samples_idx, feature])
        elif kind == 'numeric':
            # Select mean point as cut point of this feature
            cuts.append(np.mean(self.X[samples_idx, feature]))
            cuts.append(np.max(self.X[samples_idx, feature]) + 1)

        return cuts

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
            kind = self._get_feature_kind(feature)
            values = self._select_cut_points(samples_idx, feature, kind)
            for value in values:
                if kind == 'categorical':
                    n_value = sum(self.X[samples_idx, feature] == value)
                    idx = np.where(self.X[samples_idx, feature] == value)
                elif kind == 'numeric':
                    n_value = sum(self.X[samples_idx, feature] <= value)
                    idx = np.where(self.X[samples_idx, feature] <= value)
                prob_value = n_value / n_total
                if prob_value > 0:
                    feature_samples_idx = samples_idx[idx]
                    info += prob_value * self._entropy(feature_samples_idx)
        
        return info

    def _split_info(self, samples_idx, feature):
        info = 0
        n_total = len(samples_idx)
        
        kind = self._get_feature_kind(feature)
        values = self._select_cut_points(samples_idx, feature, kind)
        for value in values:
            if kind == 'categorical':
                n_value = sum(self.X[samples_idx, feature] == value)
            elif kind == 'numeric':
                n_value = sum(self.X[samples_idx, feature] <= value)    
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
        kind = self._get_feature_kind(feature)
        values = self._select_cut_points(samples_idx, feature, kind)
        for value in values:
            ginis_feature[value] = 0

            if kind == 'categorical':
                n_values = np.sum(self.X[samples_idx, feature] == value)
                idx = np.where(self.X[samples_idx, feature] == value)
            elif kind == 'numeric':
                n_values = np.sum(self.X[samples_idx, feature] <= value)
                idx = np.where(self.X[samples_idx, feature] <= value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[idx]
                aux = prob_value * self._gini(feature_samples_idx)
                ginis_feature[value] += aux

            if kind == 'categorical':
                n_values = np.sum(self.X[samples_idx, feature] != value)
                idx = np.where(self.X[samples_idx, feature] != value)
            elif kind == 'numeric':
                n_values = np.sum(self.X[samples_idx, feature] > value)
                idx = np.where(self.X[samples_idx, feature] > value)
            prob_value = n_values / n_total
            if prob_value > 0:
                feature_samples_idx = samples_idx[idx]
                aux = prob_value * self._gini(feature_samples_idx)
                ginis_feature[value] += aux
        
        return gini_all - min(ginis_feature.values())
