import dgl
import numpy as np
import torch


def parse_adjlist_2metapaths(
    adjlist,
    edge_metapath_indices,
    samples=None,
    exclude=None,
    offset=None,
    mode=None,
):
    edges = []
    nodes = set()
    result_indices = []
    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(" ")))
        nodes.add(row_parsed[0])
        if len(row_parsed) > 1:
            # sampling neighbors
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [
                            (
                                False
                                if [micro1, d1 - offset] in exclude
                                or [micro2, d2 - offset] in exclude
                                else True
                            )
                            for micro1, d1, micro2, d2 in indices[
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    else:
                        mask = [
                            (
                                False
                                if [micro1, d1 - offset] in exclude
                                or [micro2, d2 - offset] in exclude
                                else True
                            )
                            for d1, micro1, d2, micro2 in indices[
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    neighbors = np.array(row_parsed[1:])[mask]
                    result_indices.append(indices[mask])
                else:
                    neighbors = row_parsed[1:]
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = np.unique(row_parsed[1:], return_counts=True)
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(
                    np.random.choice(
                        len(row_parsed) - 1, samples, replace=False, p=p
                    )
                )
                if exclude is not None:
                    if mode == 0:
                        mask = [
                            (
                                False
                                if [micro1, d1 - offset] in exclude
                                or [micro2, d2 - offset] in exclude
                                else True
                            )
                            for micro1, d1, micro2, d2 in indices[sampled_idx][
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    else:
                        mask = [
                            (
                                False
                                if [micro1, d1 - offset] in exclude
                                or [micro2, d2 - offset] in exclude
                                else True
                            )
                            for d1, micro1, d2, micro2 in indices[sampled_idx][
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    neighbors = np.array(
                        [row_parsed[i + 1] for i in sampled_idx]
                    )[mask]
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = [row_parsed[i + 1] for i in sampled_idx]
                    result_indices.append(indices[sampled_idx])
        else:
            neighbors = [row_parsed[0]]
            indices = np.array([[row_parsed[0]] * indices.shape[1]])
            if mode == 1:
                indices += offset
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            edges.append((row_parsed[0], dst))
    mapping = {
        map_from: map_to for map_to, map_from in enumerate(sorted(nodes))
    }
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_adjlist(
    adjlist,
    edge_metapath_indices,
    samples=None,
    exclude=None,
    offset=None,
    mode=None,
):
    """
    Parse adjacency list and filter edges based on mode, sampling, and exclusions.

    :param adjlist: (list), Adjacency list where each row is a space-separated string of node indices.
    :param edge_metapath_indices : (dict), Mapping of node indices to their metapaths.
    :param samples: (int, optional), Number of neighbors to sample.
    :param exclude: (list, optional), List of edges to exclude.
    :param offset: (int, optional), Offset for disease indices.
    :param mode: (int or list, optional), Parsing mode(s) (0, 1, or 2).

    :return tuple: Parsed edges, result indices, total nodes, and mapping.
    """
    use_masks = [
        [True, True, False, False],
        [True, True, False, False],
        [False, True, True, False],
    ]

    edges = []
    nodes = set()
    result_indices = []

    for row, indices in zip(adjlist, edge_metapath_indices):
        row_parsed = list(map(int, row.split(" ")))
        (
            nodes.add(row_parsed[0])
            if mode == 0 or mode == 1
            else nodes.add(row_parsed[1])
        )

        if len(row_parsed) > 1:
            if samples is None:
                if exclude is not None:
                    if mode == 0:
                        mask = [
                            (
                                False
                                if [micro1, d1 - offset] in exclude
                                or [micro2, d2 - offset] in exclude
                                else True
                            )
                            for micro1, d1, micro2, d2 in indices[
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    elif mode == 1:
                        mask = [
                            (
                                False
                                if [d1 - offset, micro1] in exclude
                                or [d2 - offset, micro2] in exclude
                                else True
                            )
                            for d1, micro1, d2, micro2 in indices[
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    else:  # mode == 2
                        # TODO: Problem with (2, 1, 0, 1, 2)
                        if use_masks[2][2]:
                            mask = [
                                (
                                    False
                                    if (
                                        [d1 - offset, micro1] in exclude
                                        or [d2 - offset, micro2] in exclude
                                    )
                                    else True
                                )
                                for d1, micro1, d2, micro2 in indices[
                                    :, [0, 1, -2, -3]
                                ]
                            ]
                        else:
                            mask = [
                                (
                                    False
                                    if (
                                        [micro1, d1 - offset] in exclude
                                        or [micro2, d2 - offset] in exclude
                                    )
                                    else True
                                )
                                for micro1, d1, micro2, d2 in indices[
                                    :, [0, 1, -2, -3]
                                ]
                            ]
                    neighbors = (
                        np.array(row_parsed[1:])[mask]
                        if mode == 0 or mode == 1
                        else np.array(row_parsed[0:1] + row_parsed[2:])[mask]
                    )
                    result_indices.append(indices[mask])
                else:
                    neighbors = (
                        row_parsed[1:]
                        if mode == 0 or mode == 1
                        else row_parsed[0:1] + row_parsed[2:]
                    )
                    result_indices.append(indices)
            else:
                # undersampling frequent neighbors
                unique, counts = (
                    np.unique(row_parsed[1:], return_counts=True)
                    if mode == 0 or mode == 1
                    else np.unique(
                        row_parsed[0:1] + row_parsed[2:], return_counts=True
                    )
                )
                p = []
                for count in counts:
                    p += [(count ** (3 / 4)) / count] * count
                p = np.array(p)
                p = p / p.sum()
                samples = min(samples, len(row_parsed) - 1)
                sampled_idx = np.sort(
                    np.random.choice(
                        len(row_parsed) - 1, samples, replace=False, p=p
                    )
                )

                if exclude is not None:
                    if mode == 0:
                        mask = [
                            (
                                False
                                if [micro1, d1 - offset] in exclude
                                or [micro2, d2 - offset] in exclude
                                else True
                            )
                            for micro1, d1, micro2, d2 in indices[sampled_idx][
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    elif mode == 1:
                        mask = [
                            (
                                False
                                if [d1 - offset, micro1] in exclude
                                or [d2 - offset, micro2] in exclude
                                else True
                            )
                            for d1, micro1, d2, micro2 in indices[sampled_idx][
                                :, [0, 1, -1, -2]
                            ]
                        ]
                    else:  # mode == 2
                        # TODO: Problem with (2, 1, 0, 1, 2)
                        if use_masks[2][2]:
                            mask = [
                                (
                                    False
                                    if (
                                        [d1 - offset, micro1] in exclude
                                        or [d2 - offset, micro2] in exclude
                                    )
                                    else True
                                )
                                for d1, micro1, d2, micro2 in indices[
                                    :, [0, 1, -2, -3]
                                ]
                            ]
                        else:
                            mask = [
                                (
                                    False
                                    if (
                                        [micro1, d1 - offset] in exclude
                                        or [micro2, d2 - offset] in exclude
                                    )
                                    else True
                                )
                                for micro1, d1, micro2, d2 in indices[
                                    :, [0, 1, -2, -3]
                                ]
                            ]
                    neighbors = (
                        np.array([row_parsed[i + 1] for i in sampled_idx])[
                            mask
                        ]
                        if mode == 0 or mode == 1
                        else np.array(
                            [row_parsed[i] for i in sampled_idx if i != 1]
                        )[mask]
                    )
                    result_indices.append(indices[sampled_idx][mask])
                else:
                    neighbors = (
                        [row_parsed[i + 1] for i in sampled_idx]
                        if mode == 0 or mode == 1
                        else [row_parsed[i] for i in sampled_idx if i != 1]
                    )
                    result_indices.append(indices[sampled_idx])
        else:
            neighbors = (
                [row_parsed[0]] if mode == 0 or mode == 1 else [row_parsed[1]]
            )
            indices += (
                offset[0] if mode == 1 else offset[1] if mode == 2 else None
            )
            result_indices.append(indices)
        for dst in neighbors:
            nodes.add(dst)
            (
                edges.append((row_parsed[0], dst))
                if mode == 0 or mode == 1
                else edges.append((row_parsed[1], dst))
            )

    mapping = {
        map_from: map_to for map_to, map_from in enumerate(sorted(nodes))
    }
    edges = list(map(lambda tup: (mapping[tup[0]], mapping[tup[1]]), edges))
    result_indices = np.vstack(result_indices)
    return edges, result_indices, len(nodes), mapping


def parse_minibatch(
    adjlists_microdis,
    edge_metapath_indices_list_microdis,
    microbe_disease_batch,
    device,
    samples=None,
    use_masks=None,
    offset=None,
):
    g_lists = [[], []]  # list of graphs for two modes
    result_indices_lists = [[], []]  # edge indices for each mode
    idx_batch_mapped_lists = [[], []]  # mapped indices for nodes
    for mode, (adjlists, edge_metapath_indices_list) in enumerate(
        zip(adjlists_microdis, edge_metapath_indices_list_microdis)
    ):
        for adjlist, indices, use_mask in zip(
            adjlists, edge_metapath_indices_list, use_masks[mode]
        ):
            if use_mask:
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode]] for row in microbe_disease_batch],
                    [indices[row[mode]] for row in microbe_disease_batch],
                    samples,
                    microbe_disease_batch,
                    offset,
                    mode,
                )
            else:
                edges, result_indices, num_nodes, mapping = parse_adjlist(
                    [adjlist[row[mode]] for row in microbe_disease_batch],
                    [indices[row[mode]] for row in microbe_disease_batch],
                    samples,
                    offset=offset,
                    mode=mode,
                )

            g = dgl.graph(([], []))
            g.add_nodes(num_nodes)
            if len(edges) > 0:
                sorted_index = sorted(
                    range(len(edges)), key=lambda i: edges[i]
                )
                g.add_edges(
                    *list(
                        zip(
                            *[(edges[i][1], edges[i][0]) for i in sorted_index]
                        )
                    )
                )
                result_indices = torch.LongTensor(
                    result_indices[sorted_index]
                ).to(device)
            else:
                result_indices = torch.LongTensor(result_indices).to(device)
            g_lists[mode].append(g)
            result_indices_lists[mode].append(result_indices)
            idx_batch_mapped_lists[mode].append(
                np.array([mapping[row[mode]] for row in microbe_disease_batch])
            )

    return g_lists, result_indices_lists, idx_batch_mapped_lists


class IndexGenerator:
    def __init__(self, batch_size, num_data=None, indices=None, shuffle=True):
        if num_data is not None:
            self.num_data = num_data
            self.indices = np.arange(num_data)
        if indices is not None:
            self.num_data = len(indices)
            self.indices = np.copy(indices)
        self.batch_size = batch_size
        self.iter_counter = 0
        self.shuffle = shuffle
        if shuffle:
            np.random.shuffle(self.indices)

    def next(self):
        if self.num_iterations_left() <= 0:
            self.reset()
        self.iter_counter += 1
        return np.copy(
            self.indices[
                (self.iter_counter - 1)
                * self.batch_size : self.iter_counter
                * self.batch_size
            ]
        )

    def num_iterations(self):
        return int(np.ceil(self.num_data / self.batch_size))

    def num_iterations_left(self):
        return self.num_iterations() - self.iter_counter

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.iter_counter = 0
