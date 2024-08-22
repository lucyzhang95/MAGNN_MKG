import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn2, venn3


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


def plot_entity_node_degree_distribution(
    entity_counts,
    entity_name1,
    entity_name2,
    subplot_position=1,
    color="blue",
    threshold=None,
    threshold_label=None,
    save_path=None,
):
    sorted_entity_counts = entity_counts.sort_values(ascending=False)
    if len(entity_counts) < 1000:
        rounded_length = math.ceil(len(entity_counts) / 100) * 100
    else:
        rounded_length = math.ceil(len(entity_counts) / 1000) * 1000

    plt.figure(figsize=(15, 8), dpi=300)
    plt.subplot(2, 1, subplot_position)
    plt.bar(
        range(1, len(sorted_entity_counts) + 1),
        sorted_entity_counts.values,
        color=color,
        alpha=0.7,
    )
    plt.xlabel(f"{entity_name1.capitalize()} index ranges", fontsize=18)
    plt.ylabel(
        f"Node degree of {entity_name1} \n in {entity_name1.capitalize()}-{entity_name2.capitalize()}",
        fontsize=18,
    )

    plt.xticks(
        ticks=range(0, rounded_length + 1, max(1, rounded_length // 10)),
        fontsize=18,
    )
    plt.xlim(1, len(sorted_entity_counts))
    plt.yticks(fontsize=18)

    if threshold is not None:
        plt.axhline(
            y=threshold,
            color="red",
            linestyle="--",
            linewidth=2.5,
            label=threshold_label,
        )
        plt.legend(fontsize=18)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, transparent=True)

    plt.show()


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


def plot_venn_diagram(
    data_sets,
    labels,
    venn_type="venn2",
    colors=None,
    fontsize=18,
    save_path=None,
):
    plt.figure(figsize=(8, 8), dpi=300)
    if venn_type == "venn2":
        v = venn2(
            subsets=data_sets,
            set_labels=labels,
            set_colors=colors if colors else ("#1f77b4", "#ff7f0e"),
        )
    elif venn_type == "venn3":
        v = venn3(
            subsets=data_sets,
            set_labels=labels,
            set_colors=colors if colors else ("#1f77b4", "#ff7f0e", "#2ca02c"),
        )

    for text in v.set_labels:
        text.set(fontsize=fontsize)
    for text in v.subset_labels:
        text.set_fontsize(fontsize)

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, transparent=True)
    plt.show()


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


def plot_common_entity_node_degree_distribution(
    entity_counts,
    entity_name,
    color1="blue",
    color2="green",
    label1="Dataset 1",
    label2="Dataset 2",
    save_path=None,
):
    # Unpack the node degrees
    entities = [item[0] for item in entity_counts]
    counts1 = [
        item[1] for item in entity_counts
    ]  # Node degrees for the first dataset
    counts2 = [
        item[2] for item in entity_counts
    ]  # Node degrees for the second dataset

    # Sort by the sum of node degrees in descending order
    combined_counts = sorted(
        zip(entities, counts1, counts2),
        key=lambda x: x[1] + x[2],
        reverse=True,
    )
    sorted_entities, sorted_counts1, sorted_counts2 = zip(*combined_counts)

    plt.figure(figsize=(12, 6), dpi=300)

    # Plotting the bar charts for both sets of node degrees
    x = range(len(sorted_entities))
    plt.bar(x, sorted_counts1, color=color1, alpha=0.7, label=label1)
    plt.bar(
        x,
        sorted_counts2,
        bottom=sorted_counts1,  # Stacking the second bar on top of the first
        color=color2,
        alpha=0.7,
        label=label2,
    )

    plt.xlabel(f"{entity_name.capitalize()} index ranges", fontsize=14)
    plt.ylabel(
        f"Node degree of common {entity_name.capitalize()}s", fontsize=14
    )
    plt.xticks(
        ticks=range(
            0, len(sorted_entities), max(1, len(sorted_entities) // 10)
        ),
        fontsize=12,
    )
    plt.xlim(0, len(sorted_entities))
    plt.yticks(fontsize=12)
    plt.legend(fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, transparent=True)

    plt.show()
