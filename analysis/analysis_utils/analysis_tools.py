import math

import matplotlib.pyplot as plt
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


def plot_entity_distribution(
    entity_counts,
    entity_name1,
    entity_name2,
    subplot_position=1,
    color="blue",
    threshold=None,
    threshold_label=None,
    save_path=None,
):
    if len(entity_counts) < 1000:
        rounded_length = math.ceil(len(entity_counts) / 100) * 100
    else:
        rounded_length = math.ceil(len(entity_counts) / 1000) * 1000

    plt.figure(figsize=(15, 8), dpi=300)
    plt.subplot(2, 1, subplot_position)
    plt.bar(
        range(1, len(entity_counts) + 1),
        entity_counts.values,
        color=color,
        alpha=0.7,
    )
    plt.xlabel(f"Unique {entity_name1}", fontsize=18)
    plt.ylabel(
        f"Node degree of each {entity_name1} \n in {entity_name1.capitalize()}-{entity_name2.capitalize()}",
        fontsize=18,
    )

    plt.xticks(
        ticks=range(0, rounded_length + 1, max(1, rounded_length // 10)),
        fontsize=18,
    )
    plt.xlim(1, len(entity_counts))
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
        plt.savefig(save_path, format="png", dpi=300)

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
