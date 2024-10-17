import pathlib
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse


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
