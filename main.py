import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

from feature_vector import *
from sklearn.model_selection import cross_val_score

######################             MACHINE LEARNING MODEL             ######################

if __name__ == "__main__":
    # import the csv data from train_x.csv, val_x.csv and test_x.csv
    train_df = pd.read_csv("data/train_x.csv")
    val_df = pd.read_csv("data/val_x.csv")
    test_df = pd.read_csv("data/test_x.csv")

    # transform the data into a feature vector
    train_x = feature_vector(train_df["train_x"])
    val_x = feature_vector(val_df["val_x"])
    # test_x = feature_vector(test_df["test_x"])

    # # INFORMATION GAIN feature selection - better suited for correlated data
    #
    # # Calculate Information Gain using mutual_info_classif
    # info_gain_selector = SelectKBest(mutual_info_classif, k=10)
    # info_gain_selector.fit(train_x, train_df["train_y"])
    #
    # # Get the information gain scores and corresponding feature names
    # info_gain_scores = info_gain_selector.scores_
    #
    # # Print information gain scores and feature names
    # print("Information Gain Scores:")
    # for feature, score in enumerate(info_gain_scores):
    #     print(f"{feature}: {score}")

    # CHI SQUARE

    chi2_selector = SelectKBest(score_func=chi2, k="all")
    chi2_selector.fit(train_x, train_df["train_y"])

    # Get the chi-square scores and corresponding feature names
    chi2_scores = chi2_selector.scores_

    # Print chi-square scores and feature names
    print("\nChi-Square Scores:")
    for feature, score in enumerate(chi2_scores):
        print(f"{feature}: {score}")

    # convert the data into a feature vector with only the selected features
    selected_feature_indices = chi2_selector.get_support(indices=True)
    train_x = train_x[:, selected_feature_indices]

    # train the model
    logistic_classifier, bayes_classifier, svm_classifier = machine_learning_models(
        train_x, train_df["train_y"]
    )

    # test the performance of the model
    score_logistic = cross_val_score(logistic_classifier, val_x, val_df["val_y"], cv=5)
    print("logistic_score", score_logistic)
    score_svm = cross_val_score(svm_classifier, val_x, val_df["val_y"], cv=5)
    print("svm_score", score_svm)
    score_bayes = cross_val_score(bayes_classifier, val_x, val_df["val_y"], cv=5)
    print("bayes_score", score_bayes)

######################             NEURAL NETWORK MODEL             ######################
