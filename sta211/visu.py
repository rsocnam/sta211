import math
import pandas as pd
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.base import ClusterMixin
from scipy.cluster.hierarchy import dendrogram


def init_empty_object(keys1: set, keys2: set, init = 0):
    result = {}
    for i in keys1:
        for j in keys2:
            result[(i, j)] = init

    return result


def histo_count(df: pd.DataFrame, first_col: str, second_col: str, n_clusters: int, output_directory: str, **kwargs):
    grouped = df.groupby([first_col, second_col])
    grouped.count()

    main = set(df[first_col].values)
    sub = set(df[second_col].values)

    o = init_empty_object(main, sub)

    for (a, b) in grouped.groups:
        o[(a, b)] = len(grouped.groups[(a, b)])

    i = 0
    sizes = {
        2: (8, 8),
        3: (8, 12),
        4: (8, 16),
        5: (8, 20),
        6: (8, 24),
    }

    max_rows = math.ceil(len(main)/2)
    fig, axs = plt.subplots(max_rows, 2, figsize=sizes[max_rows], sharey=True, sharex=False, )
    # fig.suptitle("Répartition des familles de variables par cluster")

    for a in main:
        row = math.floor(i/2)
        col = i % 2
        i += 1
        values = []
        for b in sub:
            values.append(o[(a, b)])
        axs[row, col].bar(list(sub), values)
        axs[row, col].title.set_text("Cluster {}".format(a))
    plt.title(kwargs.get("title"))
    fig.savefig("{}/classif_variables_cl_{}.png".format(output_directory, n_clusters))
    fig.clf()
    plt.close(fig)


def correlation(df: pd.DataFrame, columns: list, output_directory: str, **kwargs):
    columns.sort()

    corr_matrix = df.get(columns).corr()
    data = list(corr_matrix.values)
    data.reverse()
    y_labels = list(columns)
    y_labels.reverse()

    colors = [
        # Reds, decreasing
        [229 / 256, 68 / 256, 0 / 256],
        [231 / 256, 88 / 256, 28 / 256],
        [234 / 256, 109 / 256, 56 / 256],
        [237 / 256, 130 / 256, 85 / 256],
        [240 / 256, 151 / 256, 113 / 256],
        [243 / 256, 171 / 256, 141 / 256],
        [246 / 256, 192 / 256, 170 / 256],
        [249 / 256, 213 / 256, 198 / 256],
        [252 / 256, 234 / 256, 226 / 256],
        # Center
        [256 / 256, 256 / 256, 256 / 256],
        [256 / 256, 256 / 256, 256 / 256],
        # Greens, increasing
        [234 / 256, 241 / 256, 233 / 256],
        [214 / 256, 228 / 256, 212 / 256],
        [194 / 256, 215 / 256, 191 / 256],
        [173 / 256, 201 / 256, 170 / 256],
        [153 / 256, 188 / 256, 148 / 256],
        [133 / 256, 175 / 256, 127 / 256],
        [112 / 256, 161 / 256, 106 / 256],
        [92 / 256, 148 / 256, 85 / 256],
        [72 / 256, 135 / 256, 64 / 256],
    ]
    c_map = ListedColormap(colors)

    ticks = list(map(lambda t: t-0.5, range(1, 1+len(columns))))
    fig, ax = plt.subplots()
    ax.set_xticklabels(columns, fontsize=7)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    bounds = [0]
    ranges = int(len(colors) / 2)
    for i in range(1, 1 + ranges):
        val = round(i/ranges, 1)
        bounds.append(val)
        bounds.insert(0, -val)

    norm = mpl.colors.BoundaryNorm(bounds, c_map.N)
    psm = ax.pcolor(data, cmap=c_map, vmin=-1, vmax=1, norm=norm)
    fig.colorbar(psm, ax=ax, ticks=bounds)
    plt.title(kwargs.get("title"))
    fig.savefig("{}/correlations.png".format(output_directory))
    fig.clf()
    plt.close(fig)


def hist_cross_categories(df: pd.DataFrame, col_src: str, col_comparison: str, output_directory: str, **kwargs):
    cat_dst = list(sorted(set(df[col_comparison])))
    clusters = list(map(lambda o: "Cl.{}".format(o), cat_dst))
    suffix = kwargs.get("suffix", "default")
    figure, axes = plt.subplots(5, 2, sharex=False, sharey=True, figsize=(12, 18))

    for i in range(0, 10):
        sub = df[df[col_src] == i][col_comparison].values
        values = []

        row = math.floor(i/2)
        col = i % 2

        for v in cat_dst:
            values.append(len(list(filter(lambda o: str(v) == str(o), sub))))
        axes[row, col].bar(clusters, values)
        axes[row, col].set_title("Chiffre {}".format(i))

    figure.savefig("{}/histogram_class_{}.png".format(output_directory, suffix), bbox_inches="tight")
    figure.clf()
    plt.title("Classification sur variables {}".format(suffix))
    plt.close(figure)


def cluster_dendogram(model: ClusterMixin, output_directory: str, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    # plt.xlabel("Number of points in node (or index of point if no parenthesis).")
    plt.savefig("{}/dendogram_variables.png".format(output_directory))
