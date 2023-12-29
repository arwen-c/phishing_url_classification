import pandas as pd
import cuml
import cudf
from feature_vector import feature_vector

from hyperopt import hp, STATUS_OK, fmin, tpe, Trials

# Define the space for hyperparameters for each model
LOGISTIC_PARAMS = {
    'penalty': hp.choice('penalty', ['l1', 'l2']),
    'dual': hp.choice('dual', [True, False]),
    'tol': hp.loguniform('tol', -10, 10),
    'C': hp.loguniform('c', -10, 10),
    'fit_intercept': hp.choice('fit_intercept', [True, False]),
    'intercept_scaling': hp.uniform('intercept_scaling', 0, 10),
    'class_weight': hp.choice('class_weight', ['balanced', None]),
    'random_state': hp.choice('random_state', [0]),
    'solver': hp.choice('solver', ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']),
    'max_iter': hp.choice('max_iter', range(100, 1000)),
}

SVM_PARAMS = {
    'svm_c': hp.uniform('svm_c', 0.1, 10),  # example for SVM
    'svm_gamma': hp.choice('svm_gamma', ['scale', 'auto'])
}

BAYES_PARAMS = {
    'bayes_alpha': hp.uniform('bayes_alpha', 0.1, 10)
}


def scale_data(x):
    return cuml.preprocessing.StandardScaler().fit_transform(x)


# Adjust the train model functions to accept params and return a dictionary for Hyperopt
def train_logistic_classifier(x, x_scaled, y, params):
    classifier = cuml.linear_model.LogisticRegression(**params).fit(x_scaled, y)
    score = classifier.score(x_scaled, y)
    print("Logistic Score:", score)
    return {'loss': -score, 'status': STATUS_OK}


def train_bayes_classifier(x, x_scaled, y, params):
    classifier = cuml.naive_bayes.MultinomialNB(alpha=params['bayes_alpha']).fit(x, y)
    score = classifier.score(x, y)
    print("Bayes Score:", score)
    return {'loss': -score, 'status': STATUS_OK}


def train_svm_classifier(x, x_scaled, y, params):
    classifier = cuml.svm.SVC(C=params['svm_c'], gamma=params['svm_gamma'], max_iter=100).fit(x_scaled, y)
    score = classifier.score(x_scaled, y)
    print("SVM Score:", score)
    return {'loss': -score, 'status': STATUS_OK}


def main():
    # import the csv data
    train_df = pd.read_csv("data/train_x.csv")

    # transform the data into a feature vector
    train_x = feature_vector(train_df["train_x"])
    train_x = cudf.DataFrame(train_x)
    train_y = cudf.Series(train_df["train_y"])

    # scale the data
    train_x_scaled = scale_data(train_x)

    # Define the trials
    trials = Trials()

    # Example of tuning Logistic Regression Hyperparameters
    best_logistic = fmin(
        fn=lambda params: train_logistic_classifier(train_x, train_x_scaled, train_y, params),
        space=LOGISTIC_PARAMS,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )
    print("Best params: ", best_logistic)


if __name__ == "__main__":
    main()
