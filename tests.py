from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.svm import LinearSVC as SKLinearSVC
from sklearn.svm import SVR as SKSVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier as SKOneVsRestClassifier
from sklearn.ensemble import VotingClassifier as SKVotingClassifier
from sklearn.ensemble import BaggingClassifier as SKBaggingClassifier
from sklearn.ensemble import RandomForestClassifier as SKRandomForestClassifier
from sklearn.datasets import make_classification, make_blobs, make_regression
import sklearn.metrics as metrics
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection

from simpleml.supervised import *
from simpleml.supervised.naive_bayes import *
from simpleml.unsupervised import *
from simpleml.model_selection import *
from simpleml.metrics import *
from simpleml.preprocessing import *
from simpleml.ensemble import *
from simpleml.plotting import *

import matplotlib.pyplot as plt


def test_linear_regression():
    X, y = make_regression(n_samples=100, n_features=1, 
                           n_informative=2, n_targets=1, noise=10.0) 

    X = StandardScaler().fit(X).transform(X)

    metric = r2_score

    model = LinearRegression()
    result = bootstrap(model, metric, X, y)
    print(f'simpleml BGD: mean={np.mean(result)}, std={np.std(result)}')

    plot_regression(X, y, model)

    model = LinearRegression(method='BFGS')
    result = bootstrap(model, metric, X, y)
    print(f'simpleml BFGS: mean={np.mean(result)}, std={np.std(result)}')

    model = SKLinearRegression()
    result = bootstrap(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_logistic_regression():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = LogisticRegression()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BGD: mean={np.mean(result)}, std={np.std(result)}')

    plot_decision_boundary(X, y, model)

    model = LogisticRegression(method='BFGS')
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BFGS: mean={np.mean(result)}, std={np.std(result)}')

    model = SKLogisticRegression()
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_support_vector_machine_classifier():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = SupportVectorMachineClassifier()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = SKLinearSVC(C=1e-10)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')

    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=3)

    X = StandardScaler().fit(X).transform(X)

    model = OneVsRestClassifier(SupportVectorMachineClassifier())
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml OVR: mean={np.mean(result)}, std={np.std(result)}')

    model = SKOneVsRestClassifier(SKLinearSVC(C=1e-10))
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn OVR: mean={np.mean(result)}, std={np.std(result)}')


def test_support_vector_machine_regressor():
    X, y = make_regression(n_samples=500, n_features=5, 
                           n_informative=5, n_targets=1) 

    X = StandardScaler().fit(X).transform(X)

    metric = r2_score

    model = SupportVectorMachineRegressor()
    result = bootstrap(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = SKSVR()
    result = bootstrap(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_KNN_classifier():
    X, y = make_blobs(n_samples=500, n_features=5, centers=2, cluster_std=5)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    for k in [3, 5, 7]:
        print(f'k={k}:')

        model = KNNClassifier(k=k)
        result = stratified_k_fold(model, metric, X, y)
        print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

        model = KNeighborsClassifier(n_neighbors=k)
        result = stratified_k_fold(model, metric, X, y)
        print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_KNN_regressor():
    X, y = make_regression(n_samples=500, n_features=5, 
                           n_informative=5, n_targets=1) 

    X = StandardScaler().fit(X).transform(X)

    metric = mean_squared_error

    for k in [3, 5, 7]:
        print(f'k={k}:')

        model = KNNRegressor(k=k)
        result = bootstrap(model, metric, X, y)
        print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

        model = KNeighborsRegressor(n_neighbors=k)
        result = bootstrap(model, metric, X, y)
        print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_decision_tree_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    # Create one artificial categorical feature
    X[:, 0] *= np.random.randint(10, size=X.shape[0])
    X[:, 0] = abs(X[:, 0].astype(np.int64))

    metric = f1_score

    model = DecisionTreeClassifier()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = SKDecisionTreeClassifier(criterion='entropy')
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')

    model = DecisionTreeClassifier(max_depth=3)
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml max_depth=3: mean={np.mean(result)}, std={np.std(result)}')

    model = SKDecisionTreeClassifier(criterion='entropy', max_depth=3)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn max_depth=3: mean={np.mean(result)}, std={np.std(result)}')


def test_show_decision_tree():
    X, y = make_classification(n_samples=10, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    # Create one artificial categorical feature
    X[:, 0] *= np.random.randint(10, size=X.shape[0])
    X[:, 0] = abs(X[:, 0].astype(np.int64))

    model = DecisionTreeClassifier()
    model.fit(X, y)
    model.show_decision_tree()


def test_shallow_neural_network():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = ShallowNeuralNetwork(alpha=1e-3, max_iter=1e4, activation='relu')
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BGD: mean={np.mean(result)}, std={np.std(result)}')

    model = ShallowNeuralNetwork(alpha=1e-3, max_iter=1e4, activation='relu', 
                                 method='L-BFGS-B')
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml L-BFGS-B: mean={np.mean(result)}, std={np.std(result)}')

    model = MLPClassifier(hidden_layer_sizes=(25,), activation='relu', 
                          solver='sgd', alpha=0.0, learning_rate='constant', 
                          learning_rate_init=1e-3, max_iter=int(1e4), tol=1e-3)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn SGD: mean={np.mean(result)}, std={np.std(result)}')
    
    model = MLPClassifier(hidden_layer_sizes=(25,), activation='relu', 
                          solver='lbfgs', alpha=0.0, learning_rate='constant', 
                          learning_rate_init=1e-3, max_iter=int(1e4), tol=1e-3)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn LBFGS: mean={np.mean(result)}, std={np.std(result)}')


def test_deep_neural_network():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = DeepNeuralNetwork(alpha=1e-3, max_iter=1e4, n_hid=(10, 5, 2), 
                              activation='relu')
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BGD: mean={np.mean(result)}, std={np.std(result)}')

    model = DeepNeuralNetwork(alpha=1e-3, max_iter=1e4, n_hid=(10, 5, 2), 
                              activation='relu', method='L-BFGS-B')
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml L-BFGS-B: mean={np.mean(result)}, std={np.std(result)}')

    model = MLPClassifier(hidden_layer_sizes=(10, 5, 2), activation='relu', 
                          solver='sgd', alpha=0.0, learning_rate='constant', 
                          learning_rate_init=1e-3, max_iter=int(1e4), tol=1e-3)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn SGD: mean={np.mean(result)}, std={np.std(result)}')
    
    model = MLPClassifier(hidden_layer_sizes=(10, 5, 2), activation='relu', 
                          solver='lbfgs', alpha=0.0, learning_rate='constant', 
                          learning_rate_init=1e-3, max_iter=int(1e4), tol=1e-3)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn LBFGS: mean={np.mean(result)}, std={np.std(result)}')


def test_gaussian_naive_bayes_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    # Create one artificial categorical feature
    X[:, 0] *= np.random.randint(10, size=X.shape[0])
    X[:, 0] = abs(X[:, 0].astype(np.int64))

    metric = f1_score

    model = GaussianNaiveBayesClassifier()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = GaussianNB()
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_bernoulli_naive_bayes_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = MinMaxScaler().fit(X).transform(X)

    metric = f1_score

    model = BernoulliNaiveBayesClassifier()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = BernoulliNB()
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_multinomial_naive_bayes_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)
    
    X = abs(X.astype(int))

    metric = f1_score

    model = MultinomialNaiveBayesClassifier()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = MultinomialNB()
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_kmeans_clustering():
    X, _ = make_blobs(n_samples=500, n_features=5, centers=3)

    X = StandardScaler().fit(X).transform(X)

    model = KMeansClustering(k=3)
    model.fit(X)
    y_pred = model.predict(X)

    plot_clustering(X, y_pred, model)


def test_principal_component_analysis():
    X, _ = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    pca = PrincipalComponentAnalysis(2)
    pca.fit(X)
    Z = pca.transform(X)

    X_approx = pca.inverse_transform(Z)
    print('Error X_approx:', sum(sum(X_approx - X)))


def test_metrics():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)
    
    X = StandardScaler().fit(X).transform(X)

    model = SKLogisticRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    y_true = y

    print('Confusion Matrix:', confusion_matrix(y_true, y_pred) == 
                            metrics.confusion_matrix(y_true, y_pred), sep='\n')
    print('Delta Accuracy:', accuracy(y_true, y_pred) - 
                             metrics.accuracy_score(y_true, y_pred))
    
    print('Delta Micro Recall:', recall(y_true, y_true, kind='micro') - 
                        metrics.recall_score(y_true, y_pred, average='micro'))
    print('Delta Micro Precision:', precision(y_true, y_pred, kind='micro') - 
                    metrics.precision_score(y_true, y_pred, average='micro'))
    print('Delta Micro F1-score:', f1_score(y_true, y_pred, kind='micro') - 
                            metrics.f1_score(y_true, y_pred, average='micro'))

    print('Delta Macro Recall:', recall(y_true, y_true, kind='macro') - 
                        metrics.recall_score(y_true, y_pred, average='macro'))
    print('Delta Macro Precision:', precision(y_true, y_pred, kind='macro') - 
                    metrics.precision_score(y_true, y_pred, average='macro'))
    print('Delta Macro F1-score:', f1_score(y_true, y_pred, kind='macro') - 
                            metrics.f1_score(y_true, y_pred, average='macro'))

    print('Delta All Recall:', recall(y_true, y_true, kind='all') - 
                            metrics.recall_score(y_true, y_pred, average=None))
    print('Delta All Precision:', precision(y_true, y_pred, kind='all') - 
                        metrics.precision_score(y_true, y_pred, average=None))
    print('Delta All F1-score:', f1_score(y_true, y_pred, kind='all') - 
                                metrics.f1_score(y_true, y_pred, average=None))

    print('Delta log loss:', log_loss_score(y_true, y_proba) - 
                             metrics.log_loss(y_true, y_proba))
    print('Delta zero one loss:', zero_one_loss(y_true, y_true) - 
                                  metrics.zero_one_loss(y_true, y_true))

    print('*' * 80)

    X, y = make_regression(n_samples=500, n_features=5, 
                           n_informative=5, n_targets=1) 
    model = SKLinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_true = y

    print('Delta mean_absolute_error:', mean_absolute_error(y_true, y_pred) - 
                                    metrics.mean_absolute_error(y_true, y_pred))
    print('Delta mean_squared_error:', mean_squared_error(y_true, y_pred) - 
                                    metrics.mean_squared_error(y_true, y_pred))
    print('Delta r2_score:', r2_score(y_true, y_pred) - 
                             metrics.r2_score(y_true, y_pred))


def test_preprocessing():
    X, _ = make_regression(n_samples=500, n_features=5, 
                           n_informative=5, n_targets=1) 

    X_min_max = MinMaxScaler().fit(X).transform(X)
    X_standard = StandardScaler().fit(X).transform(X)
    print('simpleml: MinMax:', np.min(X_min_max, axis=0), 
                               np.max(X_min_max, axis=0))
    print('simpleml: Standard:', np.mean(X_standard, axis=0), 
                                 np.std(X_standard, axis=0))

    X_min_max = preprocessing.MinMaxScaler().fit(X).transform(X)
    X_standard = preprocessing.StandardScaler().fit(X).transform(X)
    print('sklearn: MinMax:', np.min(X_min_max, axis=0), 
                              np.max(X_min_max, axis=0))
    print('sklearn: Standard:', np.mean(X_standard, axis=0), 
                                np.std(X_standard, axis=0))


def test_holdout():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    _, _, y_train, y_test = holdout(X, y)
    print('simpleml: Train:', np.bincount(y_train))
    print('simpleml: Test:', np.bincount(y_test))

    _, _, y_train, y_test = \
        model_selection.train_test_split(X, y, train_size=0.7, stratify=y)
    print('sklearn: Train:', np.bincount(y_train))
    print('sklearn: Test:', np.bincount(y_test))


def test_stratified_k_fold():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    model = SKLogisticRegression()
    metric = f1_score

    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    result = model_selection.cross_val_score(model, X, y, scoring='f1')
    print(f"sklearn: mean={np.mean(result)}, std={np.std(result)}")


def test_leave_one_out():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    model = SKLogisticRegression()
    metric = f1_score

    result = leave_one_out(model, metric, X, y)
    print(f'Result: mean={np.mean(result)}, std={np.std(result)}')


def test_bootstrap():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    model = SKLogisticRegression()
    metric = f1_score

    result = bootstrap(model, metric, X, y)
    print(f'Result: mean={np.mean(result)}, std={np.std(result)}')


def test_one_vs_rest_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = OneVsRestClassifier(LogisticRegression())
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = SKOneVsRestClassifier(SKLogisticRegression())
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_voting_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    models = [OneVsRestClassifier(LogisticRegression()), 
              GaussianNaiveBayesClassifier()]
    model = VotingClassifier(models)
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    models = [('m1', SKLogisticRegression()), ('m2', GaussianNB())]
    model = SKVotingClassifier(models)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_bagging_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = BaggingClassifier(KNeighborsClassifier(), n_models=10)
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = SKBaggingClassifier(KNeighborsClassifier(), n_estimators=10)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_random_forest_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    metric = f1_score

    model = RandomForestClassifier(n_trees=10)
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml n_trees=10: mean={np.mean(result)}, std={np.std(result)}')

    model = SKRandomForestClassifier(n_estimators=10)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn n_trees=10: mean={np.mean(result)}, std={np.std(result)}')

    model = RandomForestClassifier(n_trees=50)
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml n_trees=50: mean={np.mean(result)}, std={np.std(result)}')

    model = SKRandomForestClassifier(n_estimators=50)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn n_trees=50: mean={np.mean(result)}, std={np.std(result)}')


if __name__ == '__main__':
    tests = [
             test_linear_regression, 
             #test_logistic_regression,
             #test_support_vector_machine_classifier, 
             #test_support_vector_machine_regressor, 
             #test_shallow_neural_network, 
             #test_deep_neural_network,  
             #test_KNN_classifier, 
             #test_KNN_regressor,
             #test_decision_tree_classifier, 
             #test_show_decision_tree, 
             #test_gaussian_naive_bayes_classifier, 
             #test_bernoulli_naive_bayes_classifier, 
             #test_multinomial_naive_bayes_classifier, 
             #test_kmeans_clustering,
             #test_principal_component_analysis, 
             #test_metrics, 
             #test_preprocessing, 
             #test_holdout,
             #test_stratified_k_fold, 
             #test_leave_one_out, 
             #test_bootstrap, 
             #test_one_vs_rest_classifier, 
             #test_voting_classifier, 
             #test_bagging_classifier, 
             #test_random_forest_classifier
            ]

    for test in tests:
        name = test.__name__.replace('test_', '')
        print('*' * 80)
        print(f'Testing {name}:')
        test()
        print('*' * 80)
