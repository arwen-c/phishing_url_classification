import cuml
from hyperopt import hp, STATUS_OK, STATUS_FAIL


def scale_data(x):
    return cuml.preprocessing.StandardScaler().fit_transform(x)


LOGISTIC_REGRESSION_PARAMS = {
    "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet"]),
    "tol": hp.loguniform("tol", -10, 10),
    "C": hp.loguniform("c", -10, 10),
    "fit_intercept": hp.choice("fit_intercept", [True, False]),
    "class_weight": hp.choice("class_weight", ["balanced", None]),
    "l1_ratio": hp.uniform("l1_ratio", 0, 1),
    "solver": "qn",
}


def train_logistic_regression(train_x, train_y, val_x, val_y, params):
    try:
        classifier = cuml.LogisticRegression(**params)
        classifier.fit(train_x, train_y)
        score = classifier.score(val_x, val_y)
        return {"loss": -score, "status": STATUS_OK}
    except Exception as e:
        print(params)
        print(e)
        return {"status": STATUS_FAIL}


MBSGD_CLASSIFIER_PARAMS = {
    "loss": hp.choice("loss", ["hinge", "log", "squared_hinge", "squared_loss"]),
    "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet"]),
    "alpha": hp.loguniform("alpha", -10, 10),
    "l1_ratio": hp.uniform("l1_ratio", 0, 1),
    "fit_intercept": hp.choice("fit_intercept", [True, False]),
    "eta0": hp.choice("eta0", [0.0001, 0.001, 0.01, 0.1]),
    "power_t": hp.uniform("power_t", 0, 1),
    "learning_rate": hp.choice("learning_rate", ["constant", "invscaling", "adaptive"]),
}


def train_MBSGD_classifier(train_x, train_y, val_x, val_y, params):
    try:
        classifier = cuml.MBSGDClassifier(**params)
        classifier.fit(train_x, train_y)
        score = classifier.score(val_x, val_y)
        return {"loss": -score, "status": STATUS_OK}
    except Exception as e:
        print(params)
        print(e)
        return {"status": STATUS_FAIL}


SGD_PARAMS = {
    "loss": hp.choice("loss", ["hinge", "log", "squared_loss"]),
    "penalty": hp.choice("penalty", ["l1", "l2", "elasticnet"]),
    "alpha": hp.loguniform("alpha", -10, 10),
    "fit_intercept": hp.choice("fit_intercept", [True, False]),
    "tol": hp.loguniform("tol", -10, 10),
    "eta0": hp.choice("eta0", [0.0001, 0.001, 0.01, 0.1]),
    "power_t": hp.uniform("power_t", 0, 1),
    "learning_rate": hp.choice(
        "learning_rate", ["optimal", "constant", "invscaling", "adaptive"]
    ),
}


def train_stochastic_gradient_descent(train_x, train_y, val_x, val_y, params):
    try:
        classifier = cuml.SGD(**params)
        classifier.fit(train_x, train_y)
        score = classifier.score(val_x, val_y)
        return {"loss": -score, "status": STATUS_OK}
    except Exception as e:
        print(params)
        print(e)
        return {"status": STATUS_FAIL}


LINEAR_SVC_PARAMS = {
    "penalty": hp.choice("penalty", ["l1", "l2"]),
    "loss": hp.choice("loss", ["hinge", "squared_hinge"]),
    "fit_intercept": hp.choice("fit_intercept", [True, False]),
    "penalized_intercept": hp.choice("penalized_intercept", [True, False]),
    "class_weight": hp.choice("class_weight", ["balanced", None]),
    "C": hp.loguniform("c", -10, 10),
    "grad_tol": hp.loguniform("grad_tol", -10, 10),
    "change_tol": hp.loguniform("change_tol", -10, 10),
    "tol": hp.loguniform("tol", -10, 10),
}


def train_linear_svc(train_x, train_y, val_x, val_y, params):
    try:
        classifier = cuml.LinearSVC(**params)
        classifier.fit(train_x, train_y)
        score = classifier.score(val_x, val_y)
        return {"loss": -score, "status": STATUS_OK}
    except Exception as e:
        print(params)
        print(e)
        return {"status": STATUS_FAIL}


RANDOM_FOREST_PARAMS = {
    "n_estimators": hp.choice("n_estimators", [10, 100, 1000]),
    "bootstrap": hp.choice("bootstrap", [True, False]),
    "max_samples": hp.uniform("max_samples", 0.1, 1),
    "max_depth": hp.choice("max_depth", [10, 100, 1000]),
    "max_leaves": -1,
    "max_features": "auto",
    "n_bins": hp.choice("n_bins", [8, 16, 32, 64, 128, 256, 512, 1024]),
    "min_samples_leaf": hp.choice(
        "min_samples_leaf", [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ),
    "min_samples_split": hp.choice(
        "min_samples_split", [2, 4, 8, 16, 32, 64, 128, 256, 512]
    ),
    "min_impurity_decrease": hp.uniform("min_impurity_decrease", 0, 1),
}


def train_random_forest(train_x, train_y, val_x, val_y, params):
    try:
        classifier = cuml.RandomForestClassifier(**params)
        classifier.fit(train_x, train_y)
        score = classifier.score(val_x, val_y)
        return {"loss": -score, "status": STATUS_OK}
    except Exception as e:
        print(params)
        print(e)
        return {"status": STATUS_FAIL}


NAIVE_BAYES_PARAMS = {
    "alpha": hp.uniform("alpha", 0.1, 10),
    "fit_prior": hp.choice("fit_prior", [True, False]),
}


def train_naive_bayes(train_x, train_y, val_x, val_y, params):
    try:
        classifier = cuml.MultinomialNB(**params)
        classifier.fit(train_x, train_y)
        score = classifier.score(val_x, val_y)
        return {"loss": -score, "status": STATUS_OK}
    except Exception as e:
        print(params)
        print(e)
        return {"status": STATUS_FAIL}


NEAREST_NEIGHBORS_CLASSIFICATION_PARAMS = {
    "n_neighbors": hp.choice("n_neighbors", [2, 4, 8, 16, 32, 64, 128, 256, 512]),
    "metric": hp.choice("metric", cuml.neighbors.VALID_METRICS["brute"]),
}


def train_nearest_neighbors_classification(train_x, train_y, val_x, val_y, params):
    try:
        classifier = cuml.KNeighborsClassifier(**params)
        classifier.fit(train_x, train_y)
        score = classifier.score(val_x, val_y)
        return {"loss": -score, "status": STATUS_OK}
    except Exception as e:
        print(params)
        print(e)
        return {"status": STATUS_FAIL}
