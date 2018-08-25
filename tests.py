from sklearn.linear_model import LinearRegression as SKLinearRegression
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier as SKDecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.multiclass import OneVsRestClassifier as SKOneVsRestClassifier
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
import matplotlib.pyplot as plt


def test_linear_regression():
    X, y = make_regression(n_samples=500, n_features=5, 
                           n_informative=5, n_targets=1) 

    X = StandardScaler().fit(X).transform(X)

    metric = mean_squared_error

    model = LinearRegression()
    result = bootstrap(model, metric, X, y)
    print(f'simpleml BGD: mean={np.mean(result)}, std={np.std(result)}')

    model = LinearRegression(method='BFGS')
    result = bootstrap(model, metric, X, y)
    print(f'simpleml BFGS: mean={np.mean(result)}, std={np.std(result)}')

    model = SKLinearRegression()
    result = bootstrap(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


def test_logistic_regression():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = LogisticRegression()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BGD: mean={np.mean(result)}, std={np.std(result)}')

    model = LogisticRegression(method='BFGS')
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BFGS: mean={np.mean(result)}, std={np.std(result)}')

    model = SKLogisticRegression()
    result = stratified_k_fold(model, metric, X, y)
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


def test_decision_tree_classifier():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    # By now, only works with categorical features
    # TODO: Generalize to numeric and mixed features
    X = abs(X.astype(int))

    metric = f1_score

    model = DecisionTreeClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_true = y
    print(f'simpleml: {metric(y_true, y_pred)}')

    model = SKDecisionTreeClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_true = y
    print(f'sklearn: {metric(y_true, y_pred)}')


def test_neural_network():
    X, y = make_classification(n_samples=500, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = NeuralNetwork()
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BGD: mean={np.mean(result)}, std={np.std(result)}')

    model = NeuralNetwork(method='BFGS')
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml BFGS: mean={np.mean(result)}, std={np.std(result)}')
    
    model = MLPClassifier(hidden_layer_sizes=(25,), solver='sgd', 
                          learning_rate='constant', activation='logistic', 
                          learning_rate_init=1e-3, max_iter=int(1e3), alpha=0.0)
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


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
    X, _ = make_blobs(n_samples=500, n_features=5, centers=5)

    X = StandardScaler().fit(X).transform(X)

    model = KMeansClustering(k=5)
    model.fit(X)
    y_pred = model.predict(X)

    plt.title('KMeans clustering with k=5')
    plt.scatter(X[y_pred==0, 0], X[y_pred==0, 1], c='r', alpha=0.5)
    plt.scatter(X[y_pred==1, 0], X[y_pred==1, 1], c='g', alpha=0.5)
    plt.scatter(X[y_pred==2, 0], X[y_pred==2, 1], c='b', alpha=0.5)
    plt.scatter(X[y_pred==3, 0], X[y_pred==3, 1], c='y', alpha=0.5)
    plt.scatter(X[y_pred==4, 0], X[y_pred==4, 1], c='m', alpha=0.5)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker='o', s=120, c='k')
    plt.show()


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
    y_true = y

    print('Confusion Matrix:', confusion_matrix(y_true, y_pred) == metrics.confusion_matrix(y_true, y_pred), sep='\n')
    print('Delta Accuracy:', accuracy(y_true, y_pred) - metrics.accuracy_score(y_true, y_pred))
    
    print('Delta Micro Recall:', recall(y_true, y_true, kind='micro') - metrics.recall_score(y_true, y_pred, average='micro'))
    print('Delta Micro Precision:', precision(y_true, y_pred, kind='micro') - metrics.precision_score(y_true, y_pred, average='micro'))
    print('Delta Micro F1-score:', f1_score(y_true, y_pred, kind='micro') - metrics.f1_score(y_true, y_pred, average='micro'))

    print('Delta Macro Recall:', recall(y_true, y_true, kind='macro') - metrics.recall_score(y_true, y_pred, average='macro'))
    print('Delta Macro Precision:', precision(y_true, y_pred, kind='macro') - metrics.precision_score(y_true, y_pred, average='macro'))
    print('Delta Macro F1-score:', f1_score(y_true, y_pred, kind='macro') - metrics.f1_score(y_true, y_pred, average='macro'))

    print('Delta All Recall:', recall(y_true, y_true, kind='all') - metrics.recall_score(y_true, y_pred, average=None))
    print('Delta All Precision:', precision(y_true, y_pred, kind='all') - metrics.precision_score(y_true, y_pred, average=None))
    print('Delta All F1-score:', f1_score(y_true, y_pred, kind='all') - metrics.f1_score(y_true, y_pred, average=None))

    print('*' * 80)

    X, y = make_regression(n_samples=500, n_features=5, 
                           n_informative=5, n_targets=1) 
    model = SKLinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    y_true = y

    print('Delta mean_absolute_error:', mean_absolute_error(y_true, y_pred) - metrics.mean_absolute_error(y_true, y_pred))
    print('Delta mean_squared_error:', mean_squared_error(y_true, y_pred) - metrics.mean_squared_error(y_true, y_pred))
    print('Delta r2_score:', r2_score(y_true, y_pred) - metrics.r2_score(y_true, y_pred))


def test_preprocessing():
    X, y = make_regression(n_samples=500, n_features=5, 
                           n_informative=5, n_targets=1) 

    X_min_max = MinMaxScaler().fit(X).transform(X)
    X_standard = StandardScaler().fit(X).transform(X)
    print('simpleml: MinMax:', np.min(X_min_max, axis=0), np.max(X_min_max, axis=0))
    print('simpleml: Standard:', np.mean(X_standard, axis=0), np.std(X_standard, axis=0))

    X_min_max = preprocessing.MinMaxScaler().fit(X).transform(X)
    X_standard = preprocessing.StandardScaler().fit(X).transform(X)
    print('sklearn: MinMax:', np.min(X_min_max, axis=0), np.max(X_min_max, axis=0))
    print('sklearn: Standard:', np.mean(X_standard, axis=0), np.std(X_standard, axis=0))


def test_holdout():
    X, y = make_classification(n_samples=100, n_features=5, n_informative=5, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X_train, X_test, y_train, y_test = holdout(X, y)
    print('simpleml: Train:', np.bincount(y_train))
    print('simpleml: Test:', np.bincount(y_test))

    X_train, X_test, y_train, y_test = \
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
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    X = StandardScaler().fit(X).transform(X)

    metric = f1_score

    model = OneVsRestClassifier(LogisticRegression())
    result = stratified_k_fold(model, metric, X, y)
    print(f'simpleml: mean={np.mean(result)}, std={np.std(result)}')

    model = SKOneVsRestClassifier(SKLogisticRegression())
    result = stratified_k_fold(model, metric, X, y)
    print(f'sklearn: mean={np.mean(result)}, std={np.std(result)}')


if __name__ == '__main__':
    tests = [
             #test_linear_regression, 
             #test_logistic_regression, 
             #test_KNN_classifier, 
             #test_decision_tree_classifier, 
             test_neural_network, 
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
             #test_one_vs_rest_classifier
            ]

    for test in tests:
        name = test.__name__.replace('test_', '')
        print('*' * 80)
        print(f'Testing {name}:')
        test()
        print('*' * 80)
