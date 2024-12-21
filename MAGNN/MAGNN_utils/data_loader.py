import os

import compress_pickle as cp
import numpy as np
import scipy
import pickle


def load_compressed_pickle(file_path, compression="gzip"):
    """
    Load a compressed pickle file.

    - file_path (str): Path to the compressed pickle file.
    - compression (str): Compression type ('gzip', 'bz2', 'lzma'). Default is 'gzip'.

    Returns:
    - Loaded data from the pickle file.
    """
    try:
        data = cp.load(file_path, compression=compression)
        return data
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def load_preprocessed_data(prefix="data/preprocessed/"):
    in_file = open(prefix + "/0/0-1-0.adjlist", "r")
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + "/0/0-1-2-1-0.adjlist", "r")
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + "/0/0-2-0.adjlist", "r")
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + "/0/0-2-1-2-0.adjlist", "r")
    adjlist03 = [line.strip() for line in in_file]
    adjlist03 = adjlist03
    in_file.close()
    in_file = open(prefix + "/1/1-0-1.adjlist", "r")
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + "/1/1-0-2-0-1.adjlist", "r")
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + "/1/1-2-0-2-1.adjlist", "r")
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()
    in_file = open(prefix + "/1/1-2-1.adjlist", "r")
    adjlist13 = [line.strip() for line in in_file]
    adjlist13 = adjlist13
    in_file.close()
    in_file = open(prefix + "/2/2-0-2.adjlist", "r")
    adjlist20 = [line.strip() for line in in_file]
    adjlist20 = adjlist20
    in_file.close()
    in_file = open(prefix + "/2/2-0-1-0-2.adjlist", "r")
    adjlist21 = [line.strip() for line in in_file]
    adjlist21 = adjlist21
    in_file.close()
    in_file = open(prefix + "/2/2-1-0-1-2.adjlist", "r")
    adjlist22 = [line.strip() for line in in_file]
    adjlist22 = adjlist22
    in_file.close()
    in_file = open(prefix + "/2/2-1-2.adjlist", "r")
    adjlist23 = [line.strip() for line in in_file]
    adjlist23 = adjlist23
    in_file.close()

    in_file = prefix + "/0/0-1-0_idx.gz"
    idx00 = load_compressed_pickle(in_file)
    in_file = prefix + "/0/0-1-2-1-0_idx.gz"
    idx01 = load_compressed_pickle(in_file)
    in_file = prefix + "/0/0-2-0_idx.gz"
    idx02 = load_compressed_pickle(in_file)
    in_file = prefix + "/0/0-2-1-2-0_idx.gz"
    idx03 = load_compressed_pickle(in_file)
    in_file = prefix + "/1/1-0-1_idx.gz"
    idx10 = load_compressed_pickle(in_file)
    in_file = prefix + "/1/1-0-2-0-1_idx.gz"
    idx11 = load_compressed_pickle(in_file)
    in_file = prefix + "/1/1-2-0-2-1_idx.gz"
    idx12 = load_compressed_pickle(in_file)
    in_file = prefix + "/1/1-2-1_idx.gz"
    idx13 = load_compressed_pickle(in_file)
    in_file = prefix + "/2/2-0-2_idx.gz"
    idx20 = load_compressed_pickle(in_file)
    in_file = prefix + "/2/2-0-1-0-2_idx.gz"
    idx21 = load_compressed_pickle(in_file)
    in_file = prefix + "/2/2-1-0-1-2_idx.gz"
    idx22 = load_compressed_pickle(in_file)
    in_file = prefix + "/2/2-1-2_idx.gz"
    idx23 = load_compressed_pickle(in_file)

    adjM = scipy.sparse.load_npz(prefix + "/adjM.npz")
    type_mask = np.load(prefix + "/node_types.npy")
    train_val_test_pos_microbe_disease = np.load(
        prefix + "/microbe_disease_neg_pos_processed" + "/train_val_test_pos_microbe_disease.npz"
    )
    train_val_test_neg_microbe_disease = np.load(
        prefix + "/microbe_disease_neg_pos_processed" + "/train_val_test_neg_microbe_disease.npz"
    )

    return (
        [
            [adjlist00, adjlist01, adjlist02, adjlist03],
            [adjlist10, adjlist11, adjlist12, adjlist13],
            [adjlist20, adjlist21, adjlist22, adjlist23],
        ],
        [
            [idx00, idx01, idx02, idx03],
            [idx10, idx11, idx12, idx13],
            [idx20, idx21, idx22, idx23],
        ],
        adjM,
        type_mask,
        train_val_test_pos_microbe_disease,
        train_val_test_neg_microbe_disease,
    )


def load_preprocessed_data_2metapaths(prefix="data/sampled/preprocessed"):
    in_file = open(prefix + "/0/0-1-0.adjlist", "r")
    adjlist00 = [line.strip() for line in in_file]
    adjlist00 = adjlist00
    in_file.close()
    in_file = open(prefix + "/0/0-1-2-1-0.adjlist", "r")
    adjlist01 = [line.strip() for line in in_file]
    adjlist01 = adjlist01
    in_file.close()
    in_file = open(prefix + "/0/0-2-0.adjlist", "r")
    adjlist02 = [line.strip() for line in in_file]
    adjlist02 = adjlist02
    in_file.close()
    in_file = open(prefix + "/0/0-2-1-2-0.adjlist", "r")
    adjlist03 = [line.strip() for line in in_file]
    adjlist03 = adjlist03
    in_file.close()
    in_file = open(prefix + "/1/1-0-1.adjlist", "r")
    adjlist10 = [line.strip() for line in in_file]
    adjlist10 = adjlist10
    in_file.close()
    in_file = open(prefix + "/1/1-0-2-0-1.adjlist", "r")
    adjlist11 = [line.strip() for line in in_file]
    adjlist11 = adjlist11
    in_file.close()
    in_file = open(prefix + "/1/1-2-0-2-1.adjlist", "r")
    adjlist12 = [line.strip() for line in in_file]
    adjlist12 = adjlist12
    in_file.close()
    in_file = open(prefix + "/1/1-2-1.adjlist", "r")
    adjlist13 = [line.strip() for line in in_file]
    adjlist13 = adjlist13
    in_file.close()

    # in_file = prefix + "/0/0-1-0_idx.gz"
    # idx00 = load_compressed_pickle(in_file)
    # in_file = prefix + "/0/0-1-2-1-0_idx.gz"
    # idx01 = load_compressed_pickle(in_file)
    # in_file = prefix + "/0/0-2-0_idx.gz"
    # idx02 = load_compressed_pickle(in_file)
    # in_file = prefix + "/0/0-2-1-2-0_idx.gz"
    # idx03 = load_compressed_pickle(in_file)
    # in_file = prefix + "/1/1-0-1_idx.gz"
    # idx10 = load_compressed_pickle(in_file)
    # in_file = prefix + "/1/1-0-2-0-1_idx.gz"
    # idx11 = load_compressed_pickle(in_file)
    # in_file = prefix + "/1/1-2-0-2-1_idx.gz"
    # idx12 = load_compressed_pickle(in_file)
    # in_file = prefix + "/1/1-2-1_idx.gz"
    # idx13 = load_compressed_pickle(in_file)

    in_file = open(prefix + "/0/0-1-0_idx.pickle", "rb")
    idx00 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + "/0/0-1-2-1-0_idx.pickle", "rb")
    idx01 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + "/0/0-2-0_idx.pickle", "rb")
    idx02 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + "/0/0-2-1-2-0_idx.pickle", "rb")
    idx03 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + "/1/1-0-1_idx.pickle", "rb")
    idx10 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + "/1/1-0-2-0-1_idx.pickle", "rb")
    idx11 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + "/1/1-2-0-2-1_idx.pickle", "rb")
    idx12 = pickle.load(in_file)
    in_file.close()
    in_file = open(prefix + "/1/1-2-1_idx.pickle", "rb")
    idx13 = pickle.load(in_file)
    in_file.close()

    adjM = scipy.sparse.load_npz(prefix + "/adjM.npz")
    type_mask = np.load(prefix + "/node_types.npy")
    train_val_test_pos_microbe_disease = np.load(prefix + "/train_val_test_pos_microbe_disease.npz")
    train_val_test_neg_microbe_disease = np.load(prefix + "/train_val_test_neg_microbe_disease.npz")

    return (
        [
            [adjlist00, adjlist01, adjlist02, adjlist03],
            [adjlist10, adjlist11, adjlist12, adjlist13],
        ],
        [
            [idx00, idx01, idx02, idx03],
            [idx10, idx11, idx12, idx13],
        ],
        adjM,
        type_mask,
        train_val_test_pos_microbe_disease,
        train_val_test_neg_microbe_disease,
    )


def load_preprocessed_data2(prefix="data/preprocessed"):
    def load_adjlists(file_names, folder):
        adjlists = []
        for file_name in file_names:
            file_path = os.path.join(prefix, folder, f"{file_name}_adjlist.gz")
            adjlist = load_compressed_pickle(file_path)
            adjlist = [
                adjlist_idx for _, value in adjlist.items() if value for adjlist_idx in value
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
        prefix + "/microbe_disease_neg_pos_processed" + "/train_val_test_pos_microbe_disease.npz"
    )
    train_val_test_neg_microbe_disease = np.load(
        prefix + "/microbe_disease_neg_pos_processed" + "/train_val_test_neg_microbe_disease.npz"
    )

    return (
        [adjlists_0, adjlists_1, adjlists_2],
        [idxlists_0, idxlists_1, idxlists_2],
        adjM,
        type_mask,
        train_val_test_pos_microbe_disease,
        train_val_test_neg_microbe_disease,
    )
