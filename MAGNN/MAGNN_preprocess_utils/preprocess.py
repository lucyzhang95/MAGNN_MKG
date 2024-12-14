import os
import pathlib
from itertools import product

import compress_pickle as cp
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


def get_symmetric_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    Specifically designed to get immediate neighbor pairs of symmetrical metapath types
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param expected_metapaths: a list of expected metapaths
    :return: a list of python dictionaries, consisting of metapath-based neighbor pairs and intermediate paths
    """
    outs = []
    for metapath in expected_metapaths:
        # consider the edges of only half of the expected metapath
        # e.g., for metapath [0, 1, 2], only considers connections between metapath[0] and metapath[1]
        # e.g., for metapath [0, 1, 2], only considers metapath[0] and metapath[1], metapath[1] and metapath[2]
        mask = np.zeros(M.shape, dtype=bool)
        for i in range((len(metapath) - 1) // 2):
            temp = np.zeros(M.shape, dtype=bool)
            temp[
                np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])
            ] = True
            temp[
                np.ix_(type_mask == metapath[i + 1], type_mask == metapath[i])
            ] = True
            mask = np.logical_or(mask, temp)
        partial_g_nx = nx.from_numpy_array((M * mask).astype(int))

        # only need to consider the former half of the metapath
        # the latter half is symmetric to the former half
        # so targeting the neighbor pairs for the former half of the nodes are enough
        metapath_to_target = {}
        # e.g., in metapath [0, 1, 2, 1, 0] with type_mask=[0, 0, 1, 1, 1, 1, 2]
        # source = type_mask[0], so position[0, 1] target = type_mask[2], so position[6]
        # find all shortest single path from source to target with path length cutoff (middle way of the path)
        for source in (type_mask == metapath[0]).nonzero()[0]:
            for target in (
                type_mask == metapath[(len(metapath) - 1) // 2]
            ).nonzero()[0]:
                # check if there is a possible valid path from source to target node
                has_path = False
                single_source_paths = nx.single_source_shortest_path(
                    partial_g_nx, source, cutoff=(len(metapath) + 1) // 2 - 1
                )
                if target in single_source_paths:
                    has_path = True

                # if nx.has_path(partial_g_nx, source, target):
                if has_path:
                    shortests = [
                        p
                        for p in nx.all_shortest_paths(
                            partial_g_nx, source, target
                        )
                        if len(p) == (len(metapath) + 1) // 2
                    ]
                    if len(shortests) > 0:
                        metapath_to_target[target] = (
                            metapath_to_target.get(target, []) + shortests
                        )
        metapath_neighbor_pairs = {}
        for _, value in metapath_to_target.items():
            for p1 in value:
                for p2 in value:
                    metapath_neighbor_pairs[(p1[0], p2[0])] = (
                        metapath_neighbor_pairs.get((p1[0], p2[0]), [])
                        + [p1 + p2[-2::-1]]
                    )
        outs.append(metapath_neighbor_pairs)
    return outs


def get_asymmetric_metapath_neighbor_pairs(M, type_mask, expected_metapaths):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all nodes
    :param expected_metapaths: list of metapath sequences to follow
    :return: a list of dictionaries, each containing metapath-based neighbor pairs for each metapath
    """
    outs = []
    for metapath in expected_metapaths:
        mask = np.zeros(M.shape, dtype=bool)
        for i in range(len(metapath) - 1):
            temp = np.zeros(M.shape, dtype=bool)
            temp[
                np.ix_(type_mask == metapath[i], type_mask == metapath[i + 1])
            ] = True
            mask = np.logical_or(mask, temp)

        # Construct partial graph from the masked adjacency matrix
        partial_g_nx = nx.from_numpy_array((M * mask).astype(int))

        metapath_to_target = {}
        full_length = len(metapath) - 1

        for source in (type_mask == metapath[0]).nonzero()[0]:
            for target in (type_mask == metapath[-1]).nonzero()[0]:
                valid_paths = []

                # Find all paths from source to target with cutoff matching metapath length
                for path in nx.all_simple_paths(
                    partial_g_nx, source, target, cutoff=full_length
                ):
                    # Verify that the path matches the metapath sequence
                    if len(path) == len(metapath) and all(
                        type_mask[node] == metapath[i]
                        for i, node in enumerate(path)
                    ):
                        valid_paths.append(path)

                if valid_paths:
                    metapath_to_target[target] = (
                        metapath_to_target.get(target, []) + valid_paths
                    )

        metapath_neighbor_pairs = {}
        for _, paths in metapath_to_target.items():
            for p1 in paths:
                for p2 in paths:
                    metapath_neighbor_pairs[(p1[0], p2[0])] = (
                        metapath_neighbor_pairs.get((p1[0], p2[0]), [])
                        + [p1 + p2[-2::-1]]
                    )
        outs.append(metapath_neighbor_pairs)

    return outs


def get_networkx_graph(neighbor_pairs, type_mask, ctr_ntype):
    indices = np.where(type_mask == ctr_ntype)[0]
    idx_mapping = {}
    for i, idx in enumerate(indices):
        idx_mapping[idx] = i
    G_list = []
    for metapaths in neighbor_pairs:
        edge_count = 0
        sorted_metapaths = sorted(metapaths.items())
        G = nx.MultiDiGraph()
        G.add_nodes_from(range(len(indices)))
        for (src, dst), paths in sorted_metapaths:
            for _ in range(len(paths)):
                G.add_edge(idx_mapping[src], idx_mapping[dst])
                edge_count += 1
        G_list.append(G)
    return G_list


def get_edge_metapath_idx_array(neighbor_pairs):
    all_edge_metapath_idx_array = []
    for metapath_neighbor_pairs in neighbor_pairs:
        sorted_metapath_neighbor_pairs = sorted(
            metapath_neighbor_pairs.items()
        )
        edge_metapath_idx_array = []
        for _, paths in sorted_metapath_neighbor_pairs:
            edge_metapath_idx_array.extend(paths)
        edge_metapath_idx_array = np.array(edge_metapath_idx_array, dtype=int)
        all_edge_metapath_idx_array.append(edge_metapath_idx_array)
        print(edge_metapath_idx_array.shape)
    return all_edge_metapath_idx_array


def generate_triplet_array(relationship_dict, dtype=np.int16):
    """
    Generates a triplet array based on the relationship dictionary.
    :param relationship_dict: A dictionary where keys represent the middle entity
    e.g., disease, metabolite) and values are lists of entities related to the key (e.g., microbes, diseases)
    :param dtype: The data type of the resulting numpy array. Default is np.int16.

    :return np.ndarray: An array of triplets (entity1, key_entity, entity2).
    """
    return np.array(
        [
            (entity1, key_entity, entity2)
            for key_entity, entity_list in relationship_dict.items()
            for i, entity1 in enumerate(entity_list)
            for entity2 in entity_list[i + 1 :]
        ],
        dtype=dtype,
    )


def generate_long_relationship_array(
    relational_list: dict,
    intermediate_triplet: list,
    num_offset1=0,
    num_offset2=0,
    scaling_factor=0.2,
    dtype=np.int16,
):
    """
    Generalized function to generate arrays of relationships using a single relational list.

    :param relational_list: Mapping of entities to their relationships.
    :param intermediate_triplet: List of tuples representing intermediate triplets.
    :param num_offset1: int offset for entities in the relational list. Default is 0.
                        Offset is determined by interm_entity1 and 3 type mask. microbe entity has no offset.
    :param num_offset2: int offset to apply when generating the second set of sampled indices. Default is 0.
                        e.g., when f_entity and e_entity are not microbe, then need to apply offset.
                        Offset is determined by f_entity and e_entity type mask. microbe entity has no offset.
    :param scaling_factor: Proportion of relationships to sample.
    :param dtype: Data type for the output array.

    :return np.array: An array of relationships.
    """
    metapath_array = np.array(
        [
            (
                f_entity,
                interm_entity1,
                interm_entity2,
                interm_entity3,
                e_entity,
            )
            for interm_entity1, interm_entity2, interm_entity3 in intermediate_triplet
            if relational_list.get(interm_entity1 - num_offset1) is not None
            and relational_list.get(interm_entity3 - num_offset1) is not None
            if len(relational_list[interm_entity1 - num_offset1]) > 0
            and len(relational_list[interm_entity3 - num_offset1]) > 0
            for f_entity in (
                relational_list[interm_entity1 - num_offset1][
                    np.random.choice(
                        len(relational_list[interm_entity1 - num_offset1]),
                        int(
                            scaling_factor
                            * len(
                                relational_list[interm_entity1 - num_offset1]
                            )
                        ),
                        replace=False,
                    )
                ]
                + num_offset2
            )
            for e_entity in (
                relational_list[interm_entity3 - num_offset1][
                    np.random.choice(
                        len(relational_list[interm_entity3 - num_offset1]),
                        int(
                            scaling_factor
                            * len(
                                relational_list[interm_entity3 - num_offset1]
                            )
                        ),
                        replace=False,
                    )
                ]
                + num_offset2
            )
        ],
        dtype=dtype,
    )
    return metapath_array


def lexicographical_sort(array, sort_columns: list):
    """
    Lexicographically sorts a numpy array based on specified columns.

    :param array: Input numpy array to be sorted.
    :param sort_columns: List of column int indices to sort by in priority order.
           Sorting is performed starting from the last column in the list.

    :return np.ndarray: A lexicographically sorted array.
    """
    return array[np.lexsort([array[:, col] for col in reversed(sort_columns)])]


def process_and_save_metapath_idx_adjlist_in_batches(
    metapath_type,
    metapath_array,
    target_idx_list,
    offset,
    save_prefix,
    group_index,
    batch_size=1000,
    compression="gzip",
    process_idx=True,
    process_adjlist=True,
):
    """
    Process metapath into idx and adjlist files in batches, using pickle compression.

    :param metapath_type: tuple, the metapath type (e.g., (0, 1, 0)).
    :param metapath_array: numpy array, the pre-processed array representing the metapath edges.
    :param target_idx_list: numpy array, the indices of the targets to process.
    :param offset: int, the offset to add to the target indices.
    :param save_prefix: str, path prefix for saving the output files.
    :param group_index: int, index of the current metapath group for directory organization.
    :param batch_size: int, number of target indices to process in each batch.
    :param compression: str, compression method (e.g., 'gzip', 'bz2', 'lzma').
    :param process_idx: bool, whether to process and save the idx file.
    :param process_adjlist: bool, whether to process and save the adjlist file.
    """
    save_dir = save_prefix + "{}/".format(group_index)
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    idx_file_path = save_dir + "-".join(map(str, metapath_type)) + "_idx.pkl"
    adjlist_file_path = (
        save_dir + "-".join(map(str, metapath_type)) + "_adjlist.pkl"
    )

    if process_idx:
        if os.path.exists(idx_file_path):
            print(f"Loading existing idx file: {idx_file_path}")
            metapaths_mapping = cp.load(idx_file_path, compression=compression)
        else:
            print(f"Initializing new idx file: {idx_file_path}")
            metapaths_mapping = {}

    if process_adjlist:
        if os.path.exists(adjlist_file_path):
            print(f"Loading existing adjlist file: {adjlist_file_path}")
            adjlist_mapping = cp.load(
                adjlist_file_path, compression=compression
            )
        else:
            print(f"Initializing new adjlist file: {adjlist_file_path}")
            adjlist_mapping = {}

    left = 0
    right = 0
    for start_idx in range(0, len(target_idx_list), batch_size):
        end_idx = min(start_idx + batch_size, len(target_idx_list))
        batch_target_indices = target_idx_list[start_idx:end_idx]
        print(f"Processing batch {start_idx} to {end_idx}...")

        if process_idx:
            batch_mapping = {}
            for target_idx in batch_target_indices:
                while (
                    right < len(metapath_array)
                    and metapath_array[right, 0] == target_idx + offset
                ):
                    right += 1
                batch_mapping[target_idx.astype(np.int16)] = metapath_array[
                    left:right, ::-1
                ]
                left = right

            metapaths_mapping.update(batch_mapping)
            try:
                print(f"Saving updated idx mappings to {idx_file_path}...")
                cp.dump(
                    metapaths_mapping, idx_file_path, compression=compression
                )
                print(
                    f"Batch {start_idx}-{end_idx} saved successfully for idx"
                )
            except Exception as e:
                print(f"Failed to save idx batch {start_idx}-{end_idx}: {e}")
                raise

            del batch_mapping

        if process_adjlist:
            batch_adjlist = {}
            for target_idx in batch_target_indices:
                while (
                    right < len(metapath_array)
                    and metapath_array[right, 0] == target_idx + offset
                ):
                    right += 1
                neighbors = metapath_array[left:right, -1] - offset
                batch_adjlist[target_idx.astype(np.int16)] = neighbors.tolist()
                left = right

            adjlist_mapping.update(batch_adjlist)
            try:
                print(
                    f"Saving updated adjlist mappings to {adjlist_file_path}..."
                )
                cp.dump(
                    adjlist_mapping, adjlist_file_path, compression=compression
                )
                print(
                    f"Batch {start_idx}-{end_idx} saved successfully for adjlist"
                )
            except Exception as e:
                print(
                    f"Failed to save adjlist batch {start_idx}-{end_idx}: {e}"
                )
                raise

            del batch_adjlist

        del batch_target_indices


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
