import os

import pandas
from tqdm import tqdm

from data.load.load import load_dill


# the data used right now is coming from the following link: https://github.com/vonpower/PhishingDataset
# the data is already cleaned and ready to be used, str has been converted into numbers and the data has been split into train, validation and test set. Methodology for describing the conversion from str to numbers is described in the paper: [7] An Effective Phishing Detection Model Based on Character Level Convolutional Neural Network from URL.pdf
def convert(ds_train_raw_x,
            ds_train_raw_y,
            ds_test_raw_x,
            ds_test_raw_y,
            ds_val_raw_x,
            ds_val_raw_y,
            char_to_int):
    # swap around char_to_int dictionary
    int_to_char = {v: k for k, v in char_to_int.items()}

    ds_train_x = []
    for i, x in enumerate(tqdm(ds_train_raw_x)):
        ds_train_x.append("".join([int_to_char[y] for y in x if y != 0]))

    ds_val_x = []
    for i, x in enumerate(tqdm(ds_val_raw_x)):
        ds_val_x.append("".join([int_to_char[y] for y in x if y != 0]))

    ds_test_x = []
    for i, x in enumerate(tqdm(ds_test_raw_x)):
        ds_test_x.append("".join([int_to_char[y] for y in x if y != 0]))

    df_train = pandas.DataFrame()
    df_train["x"] = ds_train_x
    df_train["y"] = ds_train_raw_y

    df_val = pandas.DataFrame()
    df_val["x"] = ds_val_x
    df_val["y"] = ds_val_raw_y

    df_test = pandas.DataFrame()
    df_test["x"] = ds_test_x
    df_test["y"] = ds_test_raw_y

    return df_train, df_val, df_test


def main():
    X_train, y_train, X_test, y_test, X_val, y_val, char_to_int = load_dill()

    train_df, val_df, test_df = convert(X_train, y_train, X_test, y_test, X_val, y_val, char_to_int)

    p = os.path.dirname(os.path.abspath(__file__)) + "/.."

    os.makedirs(f"{p}/csv", exist_ok=True)

    train_df.to_csv(f"{p}/csv/train.csv")
    val_df.to_csv(f"{p}/csv/val.csv")
    test_df.to_csv(f"{p}/csv/test.csv")


if __name__ == "__main__":
    main()
