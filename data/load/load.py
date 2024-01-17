import os

import numpy as np


def load_csv(**kwargs):
    import pandas as pd

    p = os.path.dirname(os.path.abspath(__file__)) + "/.."

    train_df = pd.read_csv(f"{p}/csv/train.csv")
    val_df = pd.read_csv(f"{p}/csv/val.csv")
    test_df = pd.read_csv(f"{p}/csv/test.csv")

    train_x = train_df["x"]
    train_y = train_df["y"]

    val_x = val_df["x"]
    val_y = val_df["y"]

    test_x = test_df["x"]
    test_y = test_df["y"]

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_dill(**kwargs):
    import dill

    with open(f"{os.path.dirname(os.path.abspath(__file__))}/../dill/vonDataset20180426.dill", "rb") as f:
        dill_f = dill.load(f)
        ds_train_raw_x, ds_train_raw_y = dill_f["train_x"], dill_f["train_y"]
        ds_val_raw_x, ds_val_raw_y = dill_f["val_x"], dill_f["val_y"]
        ds_test_raw_x, ds_test_raw_y = dill_f["test_x"], dill_f["test_y"]
        char_to_int = dill_f["char_to_int"]

        ds_train_raw_x, ds_train_raw_y = remove_rows_only_zero(ds_train_raw_x, ds_train_raw_y)
        ds_val_raw_x, ds_val_raw_y = remove_rows_only_zero(ds_val_raw_x, ds_val_raw_y)
        ds_test_raw_x, ds_test_raw_y = remove_rows_only_zero(ds_test_raw_x, ds_test_raw_y)

        return (
            ds_train_raw_x,
            ds_train_raw_y,
            ds_test_raw_x,
            ds_test_raw_y,
            ds_val_raw_x,
            ds_val_raw_y,
            char_to_int,
        )


def remove_rows_only_zero(x, y):
    non_empty_indices = np.any(x != 0, axis=1)
    return x[non_empty_indices], y[non_empty_indices]


def load_feature_vector(k=10, score_func="chi2", *kwargs):
    import numpy as np

    p = os.path.dirname(os.path.abspath(__file__)) + "/.."

    # Loading the compressed .npz files and extracting the first array in each file
    train_x = np.load(f"{p}/feature_vector/{score_func}/{k}/train_x.npz")['arr_0']
    train_y = np.load(f"{p}/feature_vector/train_y.npz")['arr_0']
    val_x = np.load(f"{p}/feature_vector/{score_func}/{k}/val_x.npz")['arr_0']
    val_y = np.load(f"{p}/feature_vector/val_y.npz")['arr_0']
    test_x = np.load(f"{p}/feature_vector/{score_func}/{k}/test_x.npz")['arr_0']
    test_y = np.load(f"{p}/feature_vector/test_y.npz")['arr_0']

    return train_x, train_y, val_x, val_y, test_x, test_y