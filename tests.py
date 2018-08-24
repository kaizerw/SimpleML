from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification, make_blobs, make_regression
import sklearn.metrics as metrics
from simpleml.supervised import *
from simpleml.supervised.naive_bayes import *
from simpleml.unsupervised import *
from simpleml.model_selection import *
from simpleml.metrics import *
import matplotlib.pyplot as plt


def test_bernoulli_naive_bayes_classifier():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    maxX = X.max(axis=0)
    minX = X.min(axis=0)
    X = (X - minX) / (maxX - minX)

    model = BernoulliNaiveBayesClassifier(binarize=0.5)
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))


def test_gaussian_naive_bayes_classifier():
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


def test_multinomial_naive_bayes_classifier():
    X, y = make_classification(n_samples=500, n_features=100, n_informative=100, 
                               n_redundant=0, n_repeated=0, n_classes=5)
    
    X = abs(X.astype(int))

    model = MultinomialNaiveBayesClassifier()
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))


def test_naive_bayes_classifier():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    # This classifier only works with categorical features
    X = abs(X.astype(int))

    model = NaiveBayesClassifier()
    model.fit(X, y)
    
    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))


def test_decision_tree_classifier():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    # By now, only works with categorical features
    # TODO: Generalize to numeric and mixed features
    X = abs(X.astype(int))

    model = DecisionTreeClassifier()
    model.fit(X, y)

    n_samples_test = y.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))


def test_KNN_classifier():
    X, y = make_blobs(n_samples=500, n_features=10, centers=5)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    n_samples_test = X.shape[0]
    for k in [1, 3, 5, 7, 9]:
        model = KNNClassifier(k=k)
        y_pred = model.predict(X, y, X)
        print(f'Accuracy with k={k}:', sum(y_pred == y) / n_samples_test)


def test_linear_regression():
    X, y = make_regression(n_samples=500, n_features=10, 
                           n_informative=10, n_targets=1) 

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LinearRegression()
    model.fit(X, y)
    
    # plt.plot(model.costs)
    # plt.title('Costs')
    # plt.show()

    print('R2:', model.R2())


def test_logistic_regression():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LogisticRegression()
    model.fit(X, y)
    
    # plt.plot(model.costs)
    # plt.title('Costs')
    # plt.show()

    n_samples_test = X.shape[0]
    X = np.hstack((np.ones((n_samples_test, 1)), X))
    print('Accuracy:', sum(model.predict(X) == y) / n_samples_test)


def test_neural_network():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    model = NeuralNetwork()
    model.fit(X, y)

    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))

    model = MLPClassifier(hidden_layer_sizes=(25,), solver='sgd', learning_rate='constant', 
                          activation='logistic', learning_rate_init=1e-3, max_iter=int(1e3), alpha=0.0)
    model.fit(X, y)
    n_samples_test = X.shape[0]
    y_pred = model.predict(X)
    print('Accuracy:', (sum(y_pred == y) / n_samples_test))


def test_kmeans_clustering():
    X, y = make_blobs(n_samples=500, n_features=2, centers=5)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    model = KMeansClustering(k=5)
    model.fit(X)
    y = model.predict(X)

    plt.title('KMeans clustering with k=5')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='r', alpha=0.5)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='g', alpha=0.5)
    plt.scatter(X[y==2, 0], X[y==2, 1], c='b', alpha=0.5)
    plt.scatter(X[y==3, 0], X[y==3, 1], c='y', alpha=0.5)
    plt.scatter(X[y==4, 0], X[y==4, 1], c='m', alpha=0.5)
    plt.scatter(model.centroids[:, 0], model.centroids[:, 1], marker='o', s=120, c='k')
    plt.show()


def test_principal_component_analysis():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=5)

    mu = np.mean(X, axis=0)
    sigma = np.mean(X, axis=0)
    X = (X - mu) / sigma

    pca = PrincipalComponentAnalysis(2)
    pca.fit(X)
    Z = pca.transform(X)
    print('Principal Components:', Z)

    X_approx = pca.inverse_transform(Z)
    print('Error X_approx:', sum(sum(X_approx - X)))


def test_metrics():
    X, y = make_classification(n_samples=500, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    X = (X - mu) / sigma

    model = LogisticRegression()
    model.fit(X, y)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = model.predict(X)
    y_true = y

    print('Confusion Matrix:', confusion_matrix(y_true, y_pred) == metrics.confusion_matrix(y_true, y_pred), sep='\n')
    print('Accuracy:', accuracy(y_true, y_pred) - metrics.accuracy_score(y_true, y_pred))
    print('Error:', error(y_true, y_pred) - (1 - metrics.accuracy_score(y_true, y_pred)))
    print('Recall:', recall(y_true, y_true) - metrics.recall_score(y_true, y_pred))
    print('Precision:', precision(y_true, y_pred) - metrics.precision_score(y_true, y_pred))
    print('F1-score:', f1_score(y_true, y_pred) - metrics.f1_score(y_true, y_pred))

    print('*' * 80)

    X, y = make_regression(n_samples=500, n_features=10, 
                           n_informative=10, n_targets=1) 
    model = LinearRegression()
    model.fit(X, y)
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    y_pred = model.predict(X)
    y_true = y

    print('mean_absolute_error:', mean_absolute_error(y_true, y_pred) - metrics.mean_absolute_error(y_true, y_pred))
    print('mean_squared_error:', mean_squared_error(y_true, y_pred) - metrics.mean_squared_error(y_true, y_pred))
    print('r2_score:', r2_score(y_true, y_pred) - metrics.r2_score(y_true, y_pred))


def test_bootstrap():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    model = LogisticRegression()
    metric = accuracy

    result = bootstrap(model, metric, X, y)
    print(f'Result: mean={np.mean(result)}, std={np.std(result)}')


def test_leave_one_out():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    model = LogisticRegression()
    metric = accuracy

    result = leave_one_out(model, metric, X, y)
    print(f'Result: mean={np.mean(result)}, std={np.std(result)}')


def test_stratified_k_fold():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    model = LogisticRegression()
    metric = accuracy

    result = stratified_k_fold(model, metric, X, y)
    print(f'Result: mean={np.mean(result)}, std={np.std(result)}')


def test_holdout():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=10, 
                               n_redundant=0, n_repeated=0, n_classes=2)

    X_train, X_test, y_train, y_test = holdout(X, y)
    print('Train:', np.bincount(y_train))
    print('Test:', np.bincount(y_test))


if __name__ == '__main__':
    tests = [test_bernoulli_naive_bayes_classifier, 
             test_gaussian_naive_bayes_classifier, 
             test_multinomial_naive_bayes_classifier, 
             test_naive_bayes_classifier, 
             test_decision_tree_classifier, 
             test_KNN_classifier, 
             test_linear_regression, 
             test_logistic_regression, 
             test_neural_network, 
             test_kmeans_clustering, 
             test_principal_component_analysis, 
             test_metrics, 
             test_bootstrap, 
             test_leave_one_out, 
             test_stratified_k_fold, 
             test_holdout]

    for test in tests:
        name = test.__name__.replace('test_', '')
        print(f'Testing {name}:')
        test()
        print('*' * 80)
