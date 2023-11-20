import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def data_cleansing(dataframe):
    # Remove duplicates and update the dataframe
    global_dataframe = global_dataframe.drop_duplicates()

    # remove the data with missing values
    global_dataframe = global_dataframe.dropna(subset=["URL"])
    pass


# Transform the data back into string using the char_to_int dictionary
def int_to_char(x):
    res = "".join(
        [list(char_to_int.keys())[list(char_to_int.values()).index(i)] for i in x]
    )
    # remove from res the beginning of the string that had zero as a value
    res = res[res.find("h") :]  # is every string starting with 'h'?
    return res


# function that transforms the whole matrix into a list of strings
def matrix_to_list(x):
    res = []
    for i in range(x.shape[0]):
        res.append(int_to_char(x[i]))
    return res


# transformation of the data into a feature vector
# using https://www.tensorflow.org/text/tutorials/word2vec, https://pypi.org/project/tldextract/

# other data to vector transformation possibility:
# using handcrafted features
# /[4]%20PHISH-SAFE%20URL%20Features%20based%20Phishing%20Detection%20System%20using%20Machine%20Learning.pdf , [5] LexicalFeatureSelection.pdf  , [6] EfficientDeepLearningPhishingDetection.pdf


# using word detection and random word detection (really complex approach)
# /[3] NLPBasedPhishingAttack.pdf , /[2] ESWA_Sahingoz_Makale.pdf , [1] Phishing_URL_Detection_A_Real-Case_Scenario_Through_Login_URLs.pdf


def feature_vector(x):
    # handcrafted features

    # word detection and random word detection
    pass
