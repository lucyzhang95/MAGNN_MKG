import os

import numpy as np
import scipy

from MAGNN.MAGNN_utils.preprocess import load_compressed_pickle


def load_preprocessed_data(prefix="data/preprocessed/"):
    def load_adjlists(file_names, folder):
        adjlists = []
        for file_name in file_names:
            file_path = os.path.join(prefix, folder, f"{file_name}_adjlist.gz")
            adjlist = load_compressed_pickle(file_path)
            adjlist = [
                adjlist_idx
                for _, value in adjlist.items()
                if value
                for adjlist_idx in value
            ]
            adjlists.append(adjlist)
        return adjlists

    def load_indices(file_names, folder):
        indices = []
        for file_name in file_names:
            file_path = os.path.join(prefix, folder, f"{file_name}_idx.gz")
            idx = load_compressed_pickle(file_path)
            indices.append(idx)
        return indices

    files_0 = ["0-1-0", "0-1-2-1-0", "0-2-0", "0-2-1-2-0"]
    files_1 = ["1-0-1", "1-0-2-0-1", "1-2-0-2-1", "1-2-1"]
    files_2 = ["2-0-2", "2-0-1-0-2", "2-1-0-1-2", "2-1-2"]

    adjlists_0 = load_adjlists(files_0, "0")
    adjlists_1 = load_adjlists(files_1, "1")
    adjlists_2 = load_adjlists(files_2, "2")

    idxlists_0 = load_indices(files_0, "0")
    idxlists_1 = load_indices(files_1, "1")
    idxlists_2 = load_indices(files_2, "2")

    adjM = scipy.sparse.load_npz(prefix + "/adjM.npz")
    type_mask = np.load(prefix + "/node_types.npy")
    train_val_test_pos_microbe_disease = np.load(
        prefix
        + "/microbe_disease_neg_pos_processed"
        + "/train_val_test_pos_microbe_disease.npz"
    )
    train_val_test_neg_microbe_disease = np.load(
        prefix
        + "/microbe_disease_neg_pos_processed"
        + "/train_val_test_neg_microbe_disease.npz"
    )

    return (
        [adjlists_0, adjlists_1, adjlists_2],
        [idxlists_0, idxlists_1, idxlists_2],
        adjM,
        type_mask,
        train_val_test_pos_microbe_disease,
        train_val_test_neg_microbe_disease,
    )
