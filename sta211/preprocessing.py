import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def summary_linked_variables(dataframe: pd.DataFrame, min_ratio: float = 0.9):
    """
    :param dataframe: A Pandas Dataframe with quantitatives only columns
    :param min_ratio: Minimum Ratio (absolute) for correlation filtering
    :return: A generator with var1, var2, correlation between var1 & var2
    """
    df_corr = dataframe.corr()
    df_abs = df_corr.abs() >= min_ratio
    variables = df_abs.columns
    done = set()
    duplicated = set()
    for c in variables:
        done.add(c)
        todo = filter(lambda v: v not in done, variables)
        for r in todo:
            if df_abs[c][r] == True and c not in duplicated:
                yield c, r, df_corr[c][r]


def extreme_ratio_from_categorical(dataframe: pd.DataFrame, quantitatives: list, category: str, limit: int = 2):
    """
    :param dataframe: A Pandas Dataframe with normalized quantitatives columns.
    :param quantitatives: List of quantitative columns to look in dataframe
    :param category: Column of categorical Serie in dataframe
    :param limit: Limit for scaled value. Default 2.
    :return: A generator with (category, total, rows found with at least 1 column > :limit, ratio of columns with values > :limit
    """

    df_bool = dataframe[quantitatives].abs()\
        .where(cond=lambda v: v >= limit, other=np.NaN)\
        .where(cond=lambda v: np.isnan(v), other=1)

    df_bool[category] = dataframe[category]

    total = {}
    found = {}
    vars = {}

    for i in df_bool.index:
        row = df_bool.iloc[i, ]
        nb = row[quantitatives].sum()
        label = row["class"]

        if label not in total:
            total[label] = 0
            found[label] = 0
            vars[label] = []

        total[label] += 1

        if (nb > 0):
            found[label] += 1
            vars[label].append(nb)

    for label in total:
        yield label, total[label], found[label], 0 if 0 == len(vars[label]) else sum(vars[label]) / len(vars[label])


def get_pca_components(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(random_state=0)),
    ])

    pipe.fit(X, y)

    dims = len(list(filter(lambda l: l >= 1, pipe._final_estimator.explained_variance_)))
    pct = pipe._final_estimator.explained_variance_ratio_[0:dims]

    return dims, sum(pct)


def get_kmeans_clusters(X, y):
    classifier = KMeans(n_clusters=10, random_state=0, n_jobs=-1)
    classifier.fit(X, y)

    df = pd.DataFrame()
    df["cluster"] = pd.Series(data=classifier.labels_, dtype="category")
    df["class"] = pd.Series(data=y, dtype="category")

    return df
