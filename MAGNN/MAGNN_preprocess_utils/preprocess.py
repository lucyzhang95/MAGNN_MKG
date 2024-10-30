import pathlib
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
import scipy
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


def generate_paths(length):
    all_paths = product([0, 1, 2], repeat=length)

    # filter out paths where adjacent numbers are the same
    valid_paths = [
        path
        for path in all_paths
        if all(path[i] != path[i + 1] for i in range(length - 1))
    ]
    return valid_paths


def get_metapath_adjacency_matrix(adjM, type_mask, metapath):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param metapath
    :return: a list of metapath-based adjacency matrices
    """
    out_adjM = scipy.sparse.csr_matrix(
        adjM[np.ix_(type_mask == metapath[0], type_mask == metapath[1])]
    )

    # multiplication of relational adjM x relational adjM (matrix multiplication)
    # e.g., metapath = [0, 1, 0], then get the index of 0 and 1, then 1 and 0
    # adjM[0, 1] * adjM[1, 0]
    for i in range(1, len(metapath) - 1):
        out_adjM = out_adjM.dot(
            scipy.sparse.csr_matrix(
                adjM[
                    np.ix_(
                        np.where(type_mask == metapath[i])[0],
                        np.where(type_mask == metapath[i + 1])[0],
                    )
                ]
            )
        )
    return out_adjM.toarray()


def evaluate_metapath_adjacency_matrix(adjM, type_mask, possible_metapaths):
    metapath_strengths_l = []

    for metapath in possible_metapaths:
        metapath_adjM = get_metapath_adjacency_matrix(
            adjM, type_mask, metapath
        )

        metapath_adjM_sum = metapath_adjM.sum()
        max_node_pair = metapath_adjM.max()
        mean_node_pair = metapath_adjM.mean()
        density_significance = np.count_nonzero(metapath_adjM) / (
            metapath_adjM.shape[0] * metapath_adjM.shape[1]
        )

        non_zero_values = metapath_adjM[metapath_adjM > 0]
        min_node_pair = non_zero_values.min()
        non_zero_mean_strength = (
            non_zero_values.mean() if non_zero_values.size > 0 else 0
        )

        metapath_strengths = {
            "metapath": metapath,
            "sum": metapath_adjM_sum,
            "max": max_node_pair,
            "min": min_node_pair,
            "mean": mean_node_pair,
            "density": density_significance,
            "non_zero_mean": non_zero_mean_strength,
        }
        metapath_strengths_l.append(metapath_strengths)

    return metapath_strengths_l


def validate_expected_metapaths(metapaths, expected_metapaths):
    """
    Validate if each sequence of three elements in 'metapaths' exists in 'expected_metapaths'.
    Returns the original metapaths containing missing triples.

    Parameters:
    metapaths (list of lists): List of actual metapaths containing paths of length 3 and 5.
    expected_metapaths (list of tuples): List of expected metapaths containing paths of length 3 and 5.

    Returns:
    list of lists: Original metapaths from 'metapaths' that contain missing triples.
    """
    expected_triple_paths = set()

    for expected_metapath in expected_metapaths:
        if len(expected_metapath) >= 3:
            # For paths of length 3 and 5, extract triples
            for i in range(len(expected_metapath) - 2):
                expected_triple_paths.add(
                    (
                        expected_metapath[i],
                        expected_metapath[i + 1],
                        expected_metapath[i + 2],
                    )
                )

    metapaths_with_missing_triples = []
    for metapath in metapaths:
        if len(metapath) < 3:
            continue

        triples = [
            (metapath[i], metapath[i + 1], metapath[i + 2])
            for i in range(len(metapath) - 2)
        ]

        # check if any triple in this metapath is missing from expected triples
        if any(triple not in expected_triple_paths for triple in triples):
            metapaths_with_missing_triples.append(metapath)

    if metapaths_with_missing_triples:
        print(
            f"Metapath types missing the expected triple combinations are: {metapaths_with_missing_triples}"
        )
        return metapaths_with_missing_triples
    else:
        print("Metapath types have all expected triple combinations!")
    return metapaths_with_missing_triples
