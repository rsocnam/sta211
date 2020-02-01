from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline


def get_best_hyper_parameters(grid: GridSearchCV):
    max_order = max(grid.cv_results_["rank_test_score"])

    orders = list()
    for i in range(1, max_order + 1):
        for r, index in enumerate(grid.cv_results_["rank_test_score"]):
            if index == i:
                orders.append(r)

    result = []
    for o in orders:
        result.append({
            "mean_train_score": grid.cv_results_["mean_train_score"][o],
            "std_train_score": grid.cv_results_["std_train_score"][o],
            "mean_test_score": grid.cv_results_["mean_test_score"][o],
            "std_test_score": grid.cv_results_["std_test_score"][o],
            "params": grid.cv_results_["params"][o],
        })

    return result


def get_mlp():
    pipes = [
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=0)),
        ("mlp", MLPClassifier(random_state=0, verbose=0, learning_rate="adaptive"))
    ]

    grid_search = [{
        "mlp__solver": ["adam", "lbfgs"],
        "mlp__activation": ["relu", "tanh", "identity"],
        "mlp__hidden_layer_sizes": [(100,), (200,), (300,), (100, 50), (200, 100), (300, 150)],
        "pca__n_components": [77, 150, 300],
    }]

    return pipes, grid_search


def get_gradient_boosting():
    pipes = [
        ("gd", GradientBoostingClassifier(verbose=1, random_state=0, warm_start=True, init=RandomForestClassifier())),
    ]

    grid_search = [{
        "gd__learning_rate": [.00005, .0001, .001, .1],
        "gd__n_estimators": [100, 200, 300, 400],
        "gd__subsample": [1, .5, .25],
    }]

    return pipes, grid_search


def get_random_forest():
    pipes = [
        ("rf", RandomForestClassifier(random_state=0, verbose=0, oob_score=True)),
    ]

    grid_search = [{
        "rf__n_estimators": range(50, 210, 50),
        "rf__max_features": [10, "sqrt", 50, 75],
        "rf__min_samples_split": [2, 8, 16],
    }]

    return pipes, grid_search


def get_extra_trees():
    pipes = [
        ("xt", ExtraTreesClassifier(random_state=0, verbose=1, oob_score=True, bootstrap=True)),
    ]

    grid_search = [{
        "xt__n_estimators": range(25, 210, 25),
        "xt__max_features": [10, "sqrt", 50, 75],
        "xt__min_samples_split": [2, 8, 16],
    }]

    return pipes, grid_search


def get_adaboost():
    pipes = [
        ("ada", AdaBoostClassifier(random_state=0, base_estimator=RandomForestClassifier())),
    ]

    grid_search = [{
        "ada__n_estimators": range(50, 260, 50),
        "ada__learning_rate": [.001, .01, .1],
        "ada__algorithm": ["SAMME", "SAMME.R"]
    }]

    return pipes, grid_search


def get_svm():
    pipes = [
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=0)),
        ("svm", SVC(random_state=0)),
    ]

    grid_search = [{
        "svm__C": [.001, .005, .01, .02, .05, .1, .25, .5, 1],
        "svm__kernel": ["rbf", "linear", "sigmoid"],
        "pca__n_components": [77, 150, 300],
    }]

    return pipes, grid_search


def get_naive_bayes():
    pipes = [
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=0)),
        ("gnb", GaussianNB()),
    ]

    grid_search = [{
        "pca__n_components": [77, 150, 300],
    }]

    return pipes, grid_search


def get_voting_classifier():
    estimators = [
        ("rf2", RandomForestClassifier(random_state=0, n_estimators=200, warm_start=True)),
        ("svm2", Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=0, n_components=150)),
            ("svm", SVC(random_state=0, kernel="linear", C=.001)),
        ])),
        ("gnb", GaussianNB()),
        ("mlp", Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(random_state=0, n_components=150)),
            ("mlp", MLPClassifier(random_state=0, activation="tanh")),
        ])),
        ("xt", ExtraTreesClassifier(random_state=0, verbose=1, n_estimators=200, warm_start=True)),
        ("ada", AdaBoostClassifier(random_state=0, n_estimators=200, base_estimator=RandomForestClassifier())),
        ("gd", GradientBoostingClassifier(verbose=1, random_state=0, learning_rate=.01, n_estimators=200, init=RandomForestClassifier(), warm_start=True)),
    ]

    params = [{
        # "svm__svm2__C": [.001, .005, .01, .02, .05, .1, .25, .5, 1],
        # "svm__svm2__kernel": ["rbf", "linear", "sigmoid"],
        # "svm__pca__n_components": [77, 150, 300],
        #
        # "rf__rf2__n_estimators": range(25, 210, 25),
        # "rf__rf2__max_features": [10, "sqrt", 50, 75],
        # "rf__rf2__min_samples_split": [2, 8, 16],
        #
        # "mlp__mlp2__solver": ["adam", "lbfgs"],
        # "mlp__mlp2__activation": ["relu", "tanh", "identity"],
        # "mlp__mlp2__hidden_layer_sizes": [(100,), (200,), (300,), (100, 50), (200, 100), (300, 150)],
        # "mlp__pca__n_components": [77, 150, 300],
    }]

    return VotingClassifier(estimators=estimators), params
