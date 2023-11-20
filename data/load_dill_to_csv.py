import dill
import pandas
from tqdm import tqdm


# the data used right now is coming from the following link: https://github.com/vonpower/PhishingDataset
# the data is already cleaned and ready to be used, str has been converted into numbers and the data has been split into train, validation and test set. Methodology for describing the conversion from str to numbers is described in the paper: [7] An Effective Phishing Detection Model Based on Character Level Convolutional Neural Network from URL.pdf
def load_data(dill_file):
    with open(dill_file, "rb") as f:
        pickleData = dill.load(f)
        ds_train_raw_x, ds_train_raw_y = pickleData["train_x"], pickleData["train_y"]
        ds_val_raw_x, ds_val_raw_y = pickleData["val_x"], pickleData["val_y"]
        ds_test_raw_x, ds_test_raw_y = pickleData["test_x"], pickleData["test_y"]
        char_to_int = pickleData["char_to_int"]

    # swap around char_to_int dictionary
    int_to_char = {v: k for k, v in char_to_int.items()}

    ds_train_x = []
    for i, x in enumerate(tqdm(ds_train_raw_x)):
        ds_train_x.append("".join([int_to_char[y] for y in x if y != 0]))

    for i, x in enumerate(tqdm(ds_train_raw_x)):
        ds_train_x.append("".join([int_to_char[y] for y in x if y != 0]))

    ds_val_x = []
    for i, x in enumerate(tqdm(ds_val_raw_x)):
        ds_val_x.append("".join([int_to_char[y] for y in x if y != 0]))

    ds_test_x = []
    for i, x in enumerate(ds_test_raw_x):
        ds_test_x.append("".join([int_to_char[y] for y in x if y != 0]))

    df_train = pandas.DataFrame()
    df_train["train_x"] = ds_train_x
    df_train["train_y"] = ds_train_raw_y

    df_val = pandas.DataFrame()
    df_val["val_x"] = ds_val_x
    df_val["val_y"] = ds_val_raw_y

    df_test = pandas.DataFrame()
    df_test["test_x"] = ds_test_x
    df_test["test_y"] = ds_test_raw_y

    return df_train, df_val, df_test


if __name__ == "__main__":
    train_df, val_df, test_df = load_data("vonDataset20180426.dill")

    train_df.to_csv("train_x.csv")
    val_df.to_csv("val_x.csv")
    test_df.to_csv("test_x.csv")

    # transformation of the data into a feature vector

    # build different model to predict if a URL is a phishing URL or not. Begin with the most simplest ones (naive bayes and support vector machines)

    # test the models on the data, compare the performance of the different models
