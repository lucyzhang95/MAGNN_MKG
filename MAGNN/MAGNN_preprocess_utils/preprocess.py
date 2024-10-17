import pathlib
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse
from sklearn.model_selection import train_test_split


def get_column(in_f, colname1, colname2, col="col1"):
    df = pd.read_csv(in_f, sep="\t", header=None, names=[colname1, colname2])
    if col == "col1":
        return df[[colname1]]
    else:
        return df[[colname2]]


def assign_index(cols: list):
    unique_cols = pd.concat(cols, ignore_index=True)
    unique_cols = unique_cols.drop_duplicates().reset_index(drop=True)
    unique_cols["index"] = unique_cols.index
    return unique_cols


def map_index_to_relation_file(
    in_files, colname1, colname2, index_map1, index_map2
):
    df_list = [
        pd.read_csv(file, sep="\t", header=None, names=[colname1, colname2])
        for file in in_files
    ]
    df = pd.concat(df_list, ignore_index=True)

    df = df.merge(index_map1, how="left", left_on=colname1, right_on=colname1)
    df = df.merge(index_map2, how="left", left_on=colname2, right_on=colname2)

    df = df[["index_x", "index_y"]]
    df.columns = ["index_1", "index_2"]
    return df


def export_index2dat(df, out_f):
    df.to_csv(out_f, sep="\t", header=False, index=False)


def split_date(
    data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42
):
    """
    Split data into train, validation, and test sets.
    The resulting index will start from 0, which will result in off-by-one error
    Need to be careful when using these sets for downstream application
    e.g. index before splitting:[index  Microbe_idx Disease_idx], [445825  1049    307]
    index after splitting:[index  Microbe_idx Disease_idx], [445824  1049    307]
    :param data:
    :param train_ratio:
    :param val_ratio:
    :param test_ratio:
    :param random_state:
    :return:
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            "train_ratio, val_ratio, and test_ratio must sum to 1.0."
        )

    train_data, temp_data = train_test_split(
        data, test_size=(1 - train_ratio), random_state=random_state
    )

    val_size = val_ratio / (val_ratio + test_ratio)

    val_data, test_data = train_test_split(
        temp_data, test_size=(1 - val_size), random_state=random_state
    )
    return (
        train_data.index.to_numpy(),
        val_data.index.to_numpy(),
        test_data.index.to_numpy(),
    )


def save_split_data2npz(train_idx, val_idx, test_idx, out_f):
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    test_idx = np.asarray(test_idx)

    np.savez(out_f, train=train_idx, val=val_idx, test=test_idx)
