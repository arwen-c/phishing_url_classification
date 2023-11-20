import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dill
from tools import *

# the data used right now is coming from the following link: https://github.com/vonpower/PhishingDataset
# the data is already cleaned and ready to be used, str has been converted into numbers and the data has been split into train, validation and test set. Methodology for describing the conversion from str to numbers is described in the paper: [7] An Effective Phishing Detection Model Based on Character Level Convolutional Neural Network from URL.pdf
dill_file = "vonDataset20180426.dill"
with open(dill_file, "rb") as f:
    pickleData = dill.load(f)
    train_x, train_y = pickleData["train_x"], pickleData["train_y"]
    val_x, val_y = pickleData["val_x"], pickleData["val_y"]
    test_x, test_y = pickleData["test_x"], pickleData["test_y"]
    char_to_int = pickleData["char_to_int"]
print("Feature Shapes:\n")
print(
    "Train set: {}".format(train_x.shape),
    " Validation set: {}".format(val_x.shape),
    " Test set: {}".format(test_x.shape),
)

# transformation of the different URL back into text
train_x = matrix_to_list(train_x)
train_y = matrix_to_list(train_y)

# transformation of the data into a feature vector


# build different model to predict if a URL is a phishing URL or not. Begin with the most simplest ones (naive bayes and support vector machines)

# test the models on the data, compare the performance of the different models
