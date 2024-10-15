import random
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


def load_data(in_f, colname1, colname2):
    df = pd.read_csv(in_f, sep="\t", header=None, names=[colname1, colname2])
    return df


def merge_df(dfs: list):
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df = merged_df.drop_duplicates()
    print(f"Total count of edges: \n{merged_df.count()}")
    return merged_df


def filter_entity(df, colname, threshold: int | None):
    entity_counts = df[colname].value_counts().sort_index()

    if threshold is not None:
        entity_counts_less_than_threshold = entity_counts[
            entity_counts < threshold
        ]
        entity_counts_more_than_threshold = entity_counts[
            entity_counts > threshold
        ]
        entity_counts_percent = int(
            (len(entity_counts_less_than_threshold) / len(entity_counts)) * 100
        )
        print("Count of unique entity:", len(entity_counts))
        print(
            f"Count of unique entity node degree < {threshold}: "
            f"{len(entity_counts_less_than_threshold)} "
            f"({entity_counts_percent}%)"
        )
        print(
            f"Count of unique entity node degree > {threshold}: "
            f"{len(entity_counts_more_than_threshold)} "
            f"({100 - entity_counts_percent}%)"
        )
        return (
            entity_counts,
            entity_counts_less_than_threshold,
            entity_counts_more_than_threshold,
        )
    else:
        print("Count of unique entity:", len(entity_counts))
        return entity_counts


def find_optimal_threshold(df, colname, desired_ratios=None):
    if desired_ratios is None:
        desired_ratios = [(50, 50), (40, 60), (30, 70), (20, 80)]

    entity_counts = df[colname].value_counts().sort_index()
    total_entities = len(entity_counts)

    thresholds = np.unique(entity_counts.values)
    threshold_data = []

    for threshold in thresholds:
        less_than_threshold = entity_counts[entity_counts < threshold].count()
        greater_than_threshold = entity_counts[
            entity_counts >= threshold
        ].count()

        less_than_prop = less_than_threshold / total_entities * 100
        greater_than_prop = greater_than_threshold / total_entities * 100

        threshold_data.append(
            {
                "Node Degree Threshold": threshold,
                "Less Than Proportion (%)": less_than_prop,
                "Greater Than Proportion (%)": greater_than_prop,
                "Ratio": (round(less_than_prop), round(greater_than_prop)),
            }
        )

    threshold_data_df = pd.DataFrame(threshold_data)

    for desired_ratio in desired_ratios:
        threshold_data_df["Difference"] = threshold_data_df.apply(
            lambda row: abs(row["Ratio"][0] - desired_ratio[0])
            + abs(row["Ratio"][1] - desired_ratio[1]),
            axis=1,
        )
        closest_match = threshold_data_df.loc[
            threshold_data_df["Difference"].idxmin()
        ]

        if closest_match["Ratio"] == desired_ratio:
            return (
                closest_match["Threshold"],
                closest_match["Ratio"],
                threshold_data_df,
            )

        # If no exact match is found, return the closest possible ratio
    closest_match = threshold_data_df.loc[
        threshold_data_df["Difference"].idxmin()
    ]
    return (
        closest_match["Threshold"],
        closest_match["Ratio"],
        threshold_data_df,
    )


def get_common_entities(df1, df2, colname, labels):
    common_entities = set(df1[colname].unique()).intersection(
        set(df2[colname].unique())
    )
    print(
        f"There are {len(common_entities)} common {colname} entities between {labels}."
    )
    return common_entities


def calculate_common_node_degree(df1, df2, colname, common_entities):
    common_node_degree = []
    for entity in common_entities:
        node_degree1 = df1[df1[colname] == entity].shape[0]
        node_degree2 = df2[df2[colname] == entity].shape[0]
        common_node_degree.append((entity, node_degree1, node_degree2))
        common_node_degree.sort(key=lambda x: x[1] + x[2], reverse=True)
    return common_node_degree


def nodes_with_m_nbrs(G, m: int):
    nodes = set()
    for node in G.nodes():
        if len(list(G.neighbors(node))) == m:
            nodes.add(node)
    print(f"Number of nodes with {m} neighbor(s): {len(nodes)}")
    return nodes


def maximal_cliques(G, size):
    cliques = list(nx.find_cliques(G))
    mcs = [clique for clique in cliques if len(clique) == size]
    print(f"Number of maximal cliques with size {size}: {len(mcs)}")
    return mcs


def generate_subgraph(
    G,
    nodes: str | list,
    node_attr1: str,
    node_attr2: str,
    cutoff: float | None,
):
    nodes = [nodes] if isinstance(nodes, str) else nodes

    node_set = set(nodes)
    for n in nodes:
        print(f"Node: {n}")
        for nbr in G.neighbors(n):
            if cutoff is None or G.nodes[nbr][node_attr2] > cutoff:
                node_set.add(nbr)

    subgraph = G.subgraph(node_set)
    type_ct = [subgraph.nodes[n][node_attr1] for n in subgraph.nodes()]

    print(f"Node neighbor count: {Counter(type_ct)}")
    print(f"Total number of nodes: {len(list(subgraph.nodes()))}")
    print(f"Total number of edges: {len(list(subgraph.edges()))}")

    return subgraph


def find_paths_with_cutoff(
    G, source, cutoff, attr=None, neighbor_sample_size=None
):
    """
    Depth-first search algorithm to find all simple paths with a specified cutoff length.
    path length is +1 of the cutoff to include the source node
    e.g., cutoff=3, path_length=4, output: A->B->C->D ([['microbe', 'metabolite', 'microbe', 'metabolite']])
    Issue is that the function does not consider cyclic relationships.
    e.g., A->B->C->A is not considered a path of length 4.
    Another issue is that running time increases exponentially after cutoff=3, path_length=4.

    :param G:
    :param source:
    :param cutoff:
    :param attr:
    :param neighbor_sample_size:
    :return:
    """
    paths = []

    # Recursive helper function to explore paths
    def dfs(current_path, current_attr):
        if len(current_path) == cutoff + 1:  # +1 includes the source node
            paths.append(current_attr)
            return

        # Explore the last neighbors of the current path
        neighbors = list(G.neighbors(current_path[-1]))

        # if the path length is 4 and the cutoff is >= 4
        # randomly sample the neighbors of the 4th node (about to explore 5th node)
        if len(current_path) == 4 and cutoff >= 4:
            neighbors = random.sample(
                neighbors, min(len(neighbors), neighbor_sample_size)
            )

        # expand visited node into current path
        # expand the node type attribute
        for neighbor in neighbors:
            if neighbor not in current_path:
                if attr and attr in G.nodes[neighbor]:
                    dfs(
                        current_path + [neighbor],
                        current_attr + [G.nodes[neighbor][attr]],
                    )
                else:
                    dfs(current_path + [neighbor], current_attr + [None])

    if attr and attr in G.nodes[source]:
        dfs([source], [G.nodes[source][attr]])
    else:
        dfs([source], [None])

    unique_paths = [list(x) for x in set(tuple(p) for p in paths)]
    return sorted(unique_paths)
