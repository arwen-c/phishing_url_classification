import pandas as pd
from tools import *
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

    # feature selection
    ### TODO add a feature selection after completing the feature vector

    # train the model
    logistic_classifier,  bayes_classifier,svm_classifier = machine_learning_models(train_x, train_df["train_y"])

    # test the performance of the model
    score_logistic = cross_val_score(logistic_classifier, val_x, val_df["val_y"], cv=5)
    print("logistic_score", score_logistic)
    score_svm = cross_val_score(svm_classifier, val_x, val_df["val_y"], cv=5)
    print("svm_score", score_svm)
    score_bayes = cross_val_score(bayes_classifier, val_x, val_df["val_y"], cv=5)
    print("bayes_score", score_bayes)

######################             NEURAL NETWORK MODEL             ######################
