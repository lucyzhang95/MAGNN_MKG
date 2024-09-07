import math

import matplotlib.pyplot as plt
import nxviz as nv
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2, venn3


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


def plot_violin_distribution(
    entity_counts,
    entity_name,
    label1="Dataset 1",
    label2="Dataset 2",
    save_path=None,
):
    # Unpack the node degrees
    data = {
        "Entity": [item[0] for item in entity_counts],
        label1: [
            item[1] for item in entity_counts
        ],  # Node degrees for the first dataset
        label2: [
            item[2] for item in entity_counts
        ],  # Node degrees for the second dataset
    }
    df = pd.DataFrame(data)
    df_melted = df.melt(
        id_vars=["Entity"], var_name="Relationship", value_name="Node Degree"
    )

    plt.figure(figsize=(12, 6), dpi=300)
    sns.violinplot(
        x="Relationship",
        y="Node Degree",
        data=df_melted,
        density_norm="width",
        inner="quartile",
    )
    plt.ylim(0, None)
    plt.title(
        f"Distribution of Node Degrees in {entity_name.capitalize()} Relationships",
        fontsize=16,
    )
    plt.xlabel("Relationship", fontsize=14)
    plt.ylabel(f"Node Degree of {entity_name.capitalize()}", fontsize=14)

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, transparent=True)

    plt.show()


def plot_density_distribution(
    entity_counts,
    entity_name,
    label1="Dataset 1",
    label2="Dataset 2",
    save_path=None,
):
    # Unpack the node degrees
    data = {
        "Entity": [item[0] for item in entity_counts],
        label1: [
            item[1] for item in entity_counts
        ],  # Node degrees for the first dataset
        label2: [
            item[2] for item in entity_counts
        ],  # Node degrees for the second dataset
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(12, 6), dpi=300)
    sns.kdeplot(df[label1], fill=True, label=label1, color="blue", alpha=0.7)
    sns.kdeplot(df[label2], fill=True, label=label2, color="green", alpha=0.7)
    plt.title(
        f"Density Plot of Node Degrees in {entity_name.capitalize()} Relationships",
        fontsize=16,
    )
    plt.xlabel("Node Degree", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=14)

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, transparent=True)

    plt.show()


def plot_common_entity_scatter_distribution(
    entity_counts,
    entity_name="Entity",
    label1="Relationship 1",
    label2="Relationship 2",
    color1="blue",
    color2="green",
    dot_size=50,
    save_path=None,
):

    df = pd.DataFrame(
        entity_counts,
        columns=[
            entity_name,
            f"{label1} Node Degree",
            f"{label2} Node Degree",
        ],
    )

    df["Index"] = range(1, len(df) + 1)

    # Plot the scatter plot
    plt.figure(figsize=(12, 6), dpi=300)
    sns.scatterplot(
        x="Index",
        y=f"{label1} Node Degree",
        data=df,
        s=dot_size,
        color=color1,
        alpha=0.7,
        label=label1,
    )
    sns.scatterplot(
        x="Index",
        y=f"{label2} Node Degree",
        data=df,
        s=dot_size,
        color=color2,
        alpha=0.7,
        label=label2,
    )

    plt.xlabel(f"{entity_name.capitalize()} Index", fontsize=18)
    plt.ylabel(f"{entity_name.capitalize()} Node Degree", fontsize=18)
    plt.legend()

    if save_path:
        plt.savefig(save_path, format="png", dpi=300, transparent=True)

    plt.show()


def plot_circos_subgraph(
    G, node_color_by="type", group_by="type", sort_by="centrality"
):
    plt.figure(figsize=(15, 15))
    nv.circos(G, node_enc_kwargs={"radius": 1})
    plt.show()
